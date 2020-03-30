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

from collections import OrderedDict

import tensorflow as tf

from tensorforce import util
from tensorforce.core import Module, parameter_modules, tf_function
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
            self.temperature = OrderedDict()
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
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True)
            ]

        elif function == 'entropies':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True)
            ]

        elif function == 'kl_divergence':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                [
                    util.to_tensor_spec(value_spec=distribution.parameters_spec, batched=True)
                    for distribution in self.distributions.values()
                ]
            ]

        elif function == 'kl_divergences':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                [
                    util.to_tensor_spec(value_spec=distribution.parameters_spec, batched=True)
                    for distribution in self.distributions.values()
                ]
            ]

        elif function == 'kldiv_reference':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True)
            ]

        elif function == 'log_probability':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                util.to_tensor_spec(value_spec=self.actions_spec, batched=True)
            ]

        elif function == 'log_probabilities':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                util.to_tensor_spec(value_spec=self.actions_spec, batched=True)
            ]

        elif function == 'sample_actions':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec(policy=self), batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                [
                    util.to_tensor_spec(value_spec=dict(type='float', shape=()), batched=False)
                    for _ in self.actions_spec
                ]
            ]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=4)
    def act(self, states, horizons, internals, auxiliaries, return_internals):
        deterministic = self.global_tensor(name='deterministic')

        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        temperatures = OrderedDict()
        if isinstance(self.temperature, dict):
            for name in self.actions_spec:
                if name in self.temperature:
                    temperatures[name] = tf.where(
                        condition=deterministic, x=zero, y=self.temperature[name].value()
                    )
                else:
                    temperatures[name] = zero
        else:
            value = tf.where(condition=deterministic, x=zero, y=self.temperature.value())
            for name in self.actions_spec:
                temperatures[name] = value

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
