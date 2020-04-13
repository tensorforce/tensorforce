# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorforce.core import ModuleDict, parameter_modules, TensorDict, TensorSpec, tf_function, \
    tf_util
from tensorforce.core.policies import Policy


class Stochastic(Policy):
    """
    Base class for stochastic policies.

    Args:
        temperature (parameter | dict[parameter], float >= 0.0): Sampling temperature, global or
            per action (<span style="color:#00C000"><b>default</b></span>: 0.0).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, temperature=0.0, device=None, summary_labels=None, l2_regularization=None, name=None,
        states_spec=None, auxiliaries_spec=None, internals_spec=None, actions_spec=None
    ):
        super().__init__(
            device=device, summary_labels=summary_labels, l2_regularization=l2_regularization,
            name=name, states_spec=states_spec, auxiliaries_spec=auxiliaries_spec,
            actions_spec=actions_spec
        )

        # Sampling temperature
        if isinstance(temperature, dict) and \
                all(name in self.actions_spec for name in temperature):
            # Different temperature per action
            self.temperature = ModuleDict()
            for name in self.actions_spec:
                if name in temperature:
                    self.temperature[name] = self.add_module(
                        name=(name + '-temperature'), module=temperature[name],
                        modules=parameter_modules, is_trainable=False, dtype='float', min_value=0.0
                    )
        else:
            # Same temperature for all actions
            self.temperature = self.add_module(
                name='temperature', module=temperature, modules=parameter_modules,
                is_trainable=False, dtype='float', min_value=0.0
            )

    def input_signature(self, function):
        if function == 'entropy':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True)
            ]

        elif function == 'entropies':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True)
            ]

        elif function == 'kl_divergence':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True),
                self.distributions.fmap(function=(lambda x: x.parameters_spec))
                    .signature(batched=True)
            ]

        elif function == 'kl_divergences':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True),
                self.distributions.fmap(function=(lambda x: x.parameters_spec))
                    .signature(batched=True)
            ]

        elif function == 'kldiv_reference':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True)
            ]

        elif function == 'log_probability':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True),
                self.actions_spec.signature(batched=True)
            ]

        elif function == 'log_probabilities':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True),
                self.actions_spec.signature(batched=True)
            ]

        elif function == 'sample_actions':
            return [
                self.states_spec.signature(batched=True),
                TensorSpec(type='int', shape=(2,)).signature(batched=True),
                self.internals_spec.signature(batched=True),
                self.auxiliaries_spec.signature(batched=True),
                self.actions_spec.fmap(function=(lambda x: TensorSpec(type='float', shape=())))
                    .signature(batched=False)
            ]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=4)
    def act(self, states, horizons, internals, auxiliaries, deterministic, return_internals):
        zero = tf_util.constant(value=0.0, dtype='float')

        temperatures = TensorDict()
        if deterministic:
            for name in self.actions_spec:
                temperatures[name] = zero
        elif isinstance(self.temperature, dict):
            for name in self.actions_spec:
                if name in self.temperature:
                    temperatures[name] = self.temperature[name].value()
                else:
                    temperatures[name] = zero
        else:
            temperature = self.temperature.value()
            for name in self.actions_spec:
                temperatures[name] = temperature

        return self.sample_actions(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            temperatures=temperatures, return_internals=return_internals
        )

    @tf_function(num_args=5)
    def log_probability(
        self, states, horizons, internals, auxiliaries, actions, reduced, return_per_action
    ):
        log_probabilities = self.log_probabilities(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions
        )

        return self.join_value_per_action(
            values=log_probabilities, reduced=reduced, return_per_action=return_per_action
        )

    @tf_function(num_args=4)
    def entropy(self, states, horizons, internals, auxiliaries, reduced, return_per_action):
        entropies = self.entropies(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        )

        return self.join_value_per_action(
            values=entropies, reduced=reduced, return_per_action=return_per_action
        )

    @tf_function(num_args=5)
    def kl_divergence(
        self, states, horizons, internals, auxiliaries, other, reduced, return_per_action
    ):
        kl_divergences = self.kl_divergences(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            other=other
        )

        return self.join_value_per_action(
            values=kl_divergences, reduced=reduced, return_per_action=return_per_action
        )

    @tf_function(num_args=5)
    def sample_actions(
        self, states, horizons, internals, auxiliaries, temperatures, return_internals
    ):
        raise NotImplementedError

    @tf_function(num_args=5)
    def log_probabilities(self, states, horizons, internals, auxiliaries, actions):
        raise NotImplementedError

    @tf_function(num_args=4)
    def entropies(self, states, horizons, internals, auxiliaries):
        raise NotImplementedError

    @tf_function(num_args=5)
    def kl_divergences(self, states, horizons, internals, auxiliaries, other):
        raise NotImplementedError

    @tf_function(num_args=4)
    def kldiv_reference(self, states, horizons, internals, auxiliaries):
        raise NotImplementedError
