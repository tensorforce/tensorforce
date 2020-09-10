# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import tensorflow as tf

from tensorforce.core import ModuleDict, parameter_modules, SignatureDict, TensorDict, TensorSpec, \
    TensorsSpec, tf_function, tf_util
from tensorforce.core.policies import Policy


class StochasticPolicy(Policy):
    """
    Base class for stochastic policies.

    Args:
        temperature (parameter | dict[parameter], float >= 0.0): Sampling temperature, global or
            per action (<span style="color:#00C000"><b>default</b></span>: 1.0).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        kldiv_reference_spec (specification):
            <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, temperature=1.0, device=None, l2_regularization=None, name=None, states_spec=None,
        auxiliaries_spec=None, internals_spec=None, actions_spec=None, kldiv_reference_spec=None
    ):
        super().__init__(
            device=device, l2_regularization=l2_regularization, name=name, states_spec=states_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec
        )

        self.kldiv_reference_spec = kldiv_reference_spec

        # Sampling temperature
        if temperature is None:
            temperature = 1.0
        if isinstance(temperature, dict) and all(name in self.actions_spec for name in temperature):
            # Different temperature per action

            def function(name, spec):
                return self.submodule(
                    name=(name + '_temperature'), module=temperature.get(name, 0.0),
                    modules=parameter_modules, is_trainable=False, dtype='float', min_value=0.0
                )

            self.temperature = self.actions_spec.fmap(
                function=function, cls=ModuleDict, with_names=True
            )

        else:
            # Same temperature for all actions
            self.temperature = self.submodule(
                name='temperature', module=temperature, modules=parameter_modules,
                is_trainable=False, dtype='float', min_value=0.0
            )

    def input_signature(self, *, function):
        if function == 'entropy':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'entropies':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'kl_divergence':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                reference=self.distributions.fmap(
                    function=(lambda x: x.parameters_spec), cls=TensorsSpec
                ).signature(batched=True)
            )

        elif function == 'kl_divergences':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                reference=self.distributions.fmap(
                    function=(lambda x: x.parameters_spec), cls=TensorsSpec
                ).signature(batched=True)
            )

        elif function == 'kldiv_reference':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'log_probability':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True)
            )

        elif function == 'log_probabilities':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True)
            )

        elif function == 'sample':
            if isinstance(self.temperature, dict):
                temperature_spec = self.actions_spec.fmap(
                    function=(lambda _: TensorSpec(type='float', shape=()))
                )
            else:
                temperature_spec = TensorSpec(type='float', shape=())
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                temperature=temperature_spec.signature(batched=False)
            )

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'entropy':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=()).signature(batched=True)
            )

        elif function == 'entropies':
            return SignatureDict(
                singleton=self.actions_spec.fmap(function=(
                    lambda spec: TensorSpec(type='float', shape=spec.shape).signature(batched=True)
                ), cls=SignatureDict)
            )

        elif function == 'kl_divergence':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=()).signature(batched=True)
            )

        elif function == 'kl_divergences':
            return SignatureDict(
                singleton=self.actions_spec.fmap(function=(
                    lambda spec: TensorSpec(type='float', shape=spec.shape).signature(batched=True)
                ), cls=SignatureDict)
            )

        elif function == 'kldiv_reference':
            return SignatureDict(
                singleton=self.kldiv_reference_spec.signature(batched=True)
            )

        elif function == 'log_probability':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=()).signature(batched=True)
            )

        elif function == 'log_probabilities':
            return SignatureDict(
                singleton=self.actions_spec.fmap(function=(
                    lambda spec: TensorSpec(type='float', shape=spec.shape).signature(batched=True)
                ), cls=SignatureDict)
            )

        elif function == 'sample':
            return SignatureDict(
                actions=self.actions_spec.signature(batched=True),
                internals=self.internals_spec.signature(batched=True)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=5)
    def act(self, *, states, horizons, internals, auxiliaries, deterministic, independent):
        assertions = list()
        if self.config.create_tf_assertions:
            if not independent:
                false = tf_util.constant(value=False, dtype='bool')
                assertions.append(tf.debugging.assert_equal(x=deterministic, y=false))

        def fn_deterministic():
            embedding, next_internals = self.network.apply(
                x=states, horizons=horizons, internals=internals, deterministic=deterministic,
                independent=independent
            )

            def function(name, distribution):
                conditions = auxiliaries.get(name, default=TensorDict())
                parameters = distribution.parametrize(x=embedding, conditions=conditions)
                return distribution.mode(parameters=parameters)

            actions = self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

            return actions, next_internals

        def fn_sample():
            if isinstance(self.temperature, dict):
                temperature = self.temperature.fmap(function=(lambda x: x.value()), cls=TensorDict)
            else:
                temperature = self.temperature.value()

            return self.sample(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                temperature=temperature, independent=independent
            )

        with tf.control_dependencies(control_inputs=assertions):
            if independent:
                return tf.cond(pred=deterministic, true_fn=fn_deterministic, false_fn=fn_sample)
            else:
                # Always sample to record summaries
                return fn_sample()

    @tf_function(num_args=5)
    def log_probability(self, *, states, horizons, internals, auxiliaries, actions):
        log_probabilities = self.log_probabilities(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions
        )

        def function(value, spec):
            return tf.reshape(tensor=value, shape=(-1, spec.size))

        log_probabilities = log_probabilities.fmap(function=function, zip_values=self.actions_spec)
        log_probabilities = tf.concat(values=tuple(log_probabilities.values()), axis=1)

        return tf.math.reduce_sum(input_tensor=log_probabilities, axis=1)

    @tf_function(num_args=4)
    def entropy(self, *, states, horizons, internals, auxiliaries):
        entropies = self.entropies(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        )

        def function(value, spec):
            return tf.reshape(tensor=value, shape=(-1, spec.size))

        entropies = entropies.fmap(function=function, zip_values=self.actions_spec)
        entropies = tf.concat(values=tuple(entropies.values()), axis=1)

        return tf.math.reduce_mean(input_tensor=entropies, axis=1)

    @tf_function(num_args=5)
    def kl_divergence(self, *, states, horizons, internals, auxiliaries, reference):
        kl_divergences = self.kl_divergences(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            reference=reference
        )

        def function(value, spec):
            return tf.reshape(tensor=value, shape=(-1, spec.size))

        kl_divergences = kl_divergences.fmap(function=function, zip_values=self.actions_spec)
        kl_divergences = tf.concat(values=tuple(kl_divergences.values()), axis=1)

        return tf.math.reduce_mean(input_tensor=kl_divergences, axis=1)

    @tf_function(num_args=5)
    def sample(self, *, states, horizons, internals, auxiliaries, temperature, independent):
        raise NotImplementedError

    @tf_function(num_args=4)
    def entropies(self, *, states, horizons, internals, auxiliaries):
        raise NotImplementedError

    @tf_function(num_args=5)
    def kl_divergences(self, *, states, horizons, internals, auxiliaries, reference):
        raise NotImplementedError

    @tf_function(num_args=4)
    def kldiv_reference(self, *, states, horizons, internals, auxiliaries):
        raise NotImplementedError

    @tf_function(num_args=5)
    def log_probabilities(self, *, states, horizons, internals, auxiliaries, actions):
        raise NotImplementedError
