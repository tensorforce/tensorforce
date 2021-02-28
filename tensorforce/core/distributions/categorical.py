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

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules, TensorDict, TensorSpec, TensorsSpec, tf_function, \
    tf_util
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):
    """
    Categorical distribution, for discrete integer actions (specification key: `categorical`).

    Args:
        temperature_mode ("predicted" | "global"): Whether to predict the temperature via a linear
            transformation of the state embedding, or to parametrize the temperature by a separate
            set of trainable weights
            (<span style="color:#00C000"><b>default</b></span>: default temperature of 1).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, temperature_mode=None, name=None, action_spec=None, input_spec=None):
        assert action_spec.type == 'int' and action_spec.num_values is not None

        if temperature_mode is None:
            parameters_spec = TensorsSpec(
                probabilities=TensorSpec(
                    type='float', shape=(action_spec.shape + (action_spec.num_values,))
                ), logits=TensorSpec(
                    type='float', shape=(action_spec.shape + (action_spec.num_values,))
                ), action_values=TensorSpec(
                    type='float', shape=(action_spec.shape + (action_spec.num_values,))
                ), state_value=TensorSpec(type='float', shape=action_spec.shape)
            )
        else:
            parameters_spec = TensorsSpec(
                probabilities=TensorSpec(
                    type='float', shape=(action_spec.shape + (action_spec.num_values,))
                ), temperature=TensorSpec(type='float', shape=action_spec.shape),
                logits=TensorSpec(
                    type='float', shape=(action_spec.shape + (action_spec.num_values,))
                ), action_values=TensorSpec(
                    type='float', shape=(action_spec.shape + (action_spec.num_values,))
                ), state_value=TensorSpec(type='float', shape=action_spec.shape)
            )
        conditions_spec = TensorsSpec()

        super().__init__(
            name=name, action_spec=action_spec, input_spec=input_spec,
            parameters_spec=parameters_spec, conditions_spec=conditions_spec
        )

        self.temperature_mode = temperature_mode

        if self.config.enable_int_action_masking:
            self.conditions_spec['mask'] = TensorSpec(
                type='bool', shape=(self.action_spec.shape + (self.action_spec.num_values,))
            )

        num_values = self.action_spec.num_values
        if self.input_spec.rank == 1:
            # Single embedding
            self.action_values = self.submodule(
                name='action_values', module='linear', modules=layer_modules,
                size=(self.action_spec.size * num_values), initialization_scale=0.01,
                input_spec=input_spec
            )
            if self.temperature_mode == 'predicted':
                self.softplus_temperature = self.submodule(
                    name='softplus_temperature', module='linear', modules=layer_modules,
                    size=self.action_spec.size, initialization_scale=0.01,
                    input_spec=self.input_spec
                )

        else:
            # Embedding per action
            if self.input_spec.rank < 1 or self.input_spec.rank > 3:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.input_spec.shape,
                    hint='invalid rank'
                )
            if self.input_spec.shape[:-1] == self.action_spec.shape[:-1]:
                size = self.action_spec.shape[-1]
            elif self.input_spec.shape[:-1] == self.action_spec.shape:
                size = 1
            else:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.input_spec.shape,
                    hint='not flattened and incompatible with action shape'
                )
            self.action_values = self.submodule(
                name='action_values', module='linear', modules=layer_modules,
                size=(size * num_values), initialization_scale=0.01, input_spec=input_spec
            )
            if self.temperature_mode == 'predicted':
                self.softplus_temperature = self.submodule(
                    name='softplus_temperature', module='linear', modules=layer_modules, size=size,
                    initialization_scale=0.01, input_spec=self.input_spec
                )

    def initialize(self):
        super().initialize()

        if self.temperature_mode == 'global':
            spec = TensorSpec(type='float', shape=((1,) + self.action_spec.shape + (1,)))
            self.softplus_temperature = self.variable(
                name='softplus_temperature', spec=spec, initializer='zeros', is_trainable=True,
                is_saved=True
            )

        prefix = 'distributions/' + self.name + '-probability'
        names = [prefix + str(n) for n in range(self.action_spec.num_values)]
        self.register_summary(label='distribution', name=names)

        spec = self.parameters_spec['probabilities']
        self.register_tracking(label='distribution', name='probabilities', spec=spec)

        if self.temperature_mode is not None:
            spec = self.parameters_spec['temperature']
            self.register_tracking(label='distribution', name='temperature', spec=spec)

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        log_epsilon = tf_util.constant(value=np.log(util.epsilon), dtype='float')
        log_two = tf_util.constant(value=np.log(2.0), dtype='float')

        # Action values
        action_values = self.action_values.apply(x=x)
        shape = (-1,) + self.action_spec.shape + (self.action_spec.num_values,)
        action_values = tf.reshape(tensor=action_values, shape=shape)

        # Softplus standard deviation
        if self.temperature_mode == 'global':
            multiples = (tf.shape(input=x)[0],) + tuple(1 for _ in range(self.action_spec.rank + 1))
            softplus_temperature = tf.tile(input=self.softplus_temperature, multiples=multiples)
        elif self.temperature_mode == 'predicted':
            softplus_temperature = self.softplus_temperature.apply(x=x)
            shape = (-1,) + self.action_spec.shape + (1,)
            softplus_temperature = tf.reshape(tensor=softplus_temperature, shape=shape)

        if self.temperature_mode is None:
            # Logits
            logits = action_values

            # Implicit states value
            state_value = tf.reduce_logsumexp(input_tensor=logits, axis=-1)

        else:
            # Clip softplus_temperature for numerical stability (epsilon < 1.0, hence negative)
            softplus_temperature = tf.clip_by_value(
                t=softplus_temperature, clip_value_min=log_epsilon, clip_value_max=-log_epsilon
            )

            # Softplus transformation (based on https://arxiv.org/abs/2007.06059)
            softplus_shift = tf_util.constant(value=0.2, dtype='float')
            temperature = (tf.nn.softplus(features=softplus_temperature) + softplus_shift) / \
                (log_two + softplus_shift)

            # Logits
            logits = action_values / temperature

            # Implicit states value
            temperature = tf.squeeze(input=temperature, axis=-1)
            state_value = temperature * tf.reduce_logsumexp(input_tensor=logits, axis=-1)

        # # Explicit states value and advantage-based action values
        # state_value = self.state_value.apply(x=x)
        # state_value = tf.reshape(tensor=state_value, shape=shape[:-1])
        # action_values = tf.expand_dims(input=state_value, axis=-1) + action_values
        # action_values -= tf.math.reduce_mean(input_tensor=action_values, axis=-1, keepdims=True)

        # Action masking, affects action_values/probabilities/logits but not state_value
        if self.config.enable_int_action_masking:
            min_float = tf.fill(
                dims=tf.shape(input=action_values), value=tf_util.get_dtype(type='float').min
            )
            action_values = tf.where(condition=conditions['mask'], x=action_values, y=min_float)
            logits = tf.where(condition=conditions['mask'], x=logits, y=min_float)

        # Softmax for corresponding probabilities
        probabilities = tf.nn.softmax(logits=logits, axis=-1)

        # "Normalized" logits
        logits = tf.math.log(x=tf.maximum(x=probabilities, y=epsilon))
        # Unstable
        # logits = tf.nn.log_softmax(logits=logits, axis=-1)
        # Doesn't take masking into account
        # logits = action_values - tf.expand_dims(input=state_value, axis=-1) ... / temperature

        if self.temperature_mode is None:
            return TensorDict(
                probabilities=probabilities, logits=logits, action_values=action_values,
                state_value=state_value
            )
        else:
            return TensorDict(
                probabilities=probabilities, temperature=temperature, logits=logits,
                action_values=action_values, state_value=state_value
            )

    @tf_function(num_args=1)
    def mode(self, *, parameters, independent):
        if self.temperature_mode is None:
            probabilities, action_values = parameters.get(('probabilities', 'action_values'))
        else:
            probabilities, temperature, action_values = parameters.get(
                ('probabilities', 'temperature', 'action_values')
            )

        # Distribution parameter summaries
        dependencies = list()
        if not independent:
            def fn_summary():
                axis = range(self.action_spec.rank + 1)
                probs = tf.math.reduce_mean(input_tensor=probabilities, axis=axis)
                probs = [probs[n] for n in range(self.action_spec.num_values)]
                if self.temperature_mode is not None:
                    probs.append(tf.math.reduce_mean(input_tensor=temperature, axis=axis))
                return probs

            prefix = 'distributions/' + self.name + '-probability'
            names = [prefix + str(n) for n in range(self.action_spec.num_values)]
            names.append('distributions/' + self.name + '-temperature')
            dependencies.extend(self.summary(
                label='distribution', name=names, data=fn_summary, step='timesteps'
            ))

        # Distribution parameter tracking
        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=probabilities, axis=0)

        dependencies.extend(
            self.track(label='distribution', name='probabilities', data=fn_tracking)
        )

        if self.temperature_mode is not None:

            def fn_tracking():
                return tf.math.reduce_mean(input_tensor=temperature, axis=0)

            dependencies.extend(
                self.track(label='distribution', name='temperature', data=fn_tracking)
            )

        with tf.control_dependencies(control_inputs=dependencies):
            action = tf.math.argmax(input=action_values, axis=-1)
            return tf_util.cast(x=action, dtype='int')

    @tf_function(num_args=2)
    def sample(self, *, parameters, temperature, independent):
        if self.temperature_mode is None:
            probabilities, logits, action_values = parameters.get(
                ('probabilities', 'logits', 'action_values')
            )
        else:
            probabilities, temp, logits, action_values = parameters.get(
                ('probabilities', 'temperature', 'logits', 'action_values')
            )

        # Distribution parameter summaries
        dependencies = list()
        if not independent:
            def fn_summary():
                axis = range(self.action_spec.rank + 1)
                probs = tf.math.reduce_mean(input_tensor=probabilities, axis=axis)
                probs = [probs[n] for n in range(self.action_spec.num_values)]
                if self.temperature_mode is not None:
                    probs.append(tf.math.reduce_mean(input_tensor=temp, axis=axis))
                return probs

            prefix = 'distributions/' + self.name + '-probability'
            names = [prefix + str(n) for n in range(self.action_spec.num_values)]
            names.append('distributions/' + self.name + '-temperature')
            dependencies.extend(self.summary(
                label='distribution', name=names, data=fn_summary, step='timesteps'
            ))

        # Distribution parameter tracking
        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=probabilities, axis=0)

        dependencies.extend(
            self.track(label='distribution', name='probabilities', data=fn_tracking)
        )

        if self.temperature_mode is not None:
            def fn_tracking():
                return tf.math.reduce_mean(input_tensor=temp, axis=0)

            dependencies.extend(
                self.track(label='distribution', name='temperature', data=fn_tracking)
            )

        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        def fn_mode():
            # Deterministic: maximum likelihood action
            action = tf.math.argmax(input=action_values, axis=-1)
            return tf_util.cast(x=action, dtype='int')

        def fn_sample():
            # Set logits to minimal value
            min_float = tf.fill(dims=tf.shape(input=logits), value=tf_util.get_dtype(type='float').min)
            temp_logits = logits / tf.math.maximum(x=temperature, y=epsilon)
            temp_logits = tf.where(condition=(probabilities < epsilon), x=min_float, y=temp_logits)

            # Non-deterministic: sample action using Gumbel distribution
            one = tf_util.constant(value=1.0, dtype='float')
            uniform_distribution = tf.random.uniform(
                shape=tf.shape(input=temp_logits), minval=epsilon, maxval=(one - epsilon),
                dtype=tf_util.get_dtype(type='float')
            )
            # Second log numerically stable since log(1-eps) ~ -eps
            gumbel_distribution = -tf.math.log(x=-tf.math.log(x=uniform_distribution))
            action = tf.math.argmax(input=(temp_logits + gumbel_distribution), axis=-1)
            return tf_util.cast(x=action, dtype='int')

        with tf.control_dependencies(control_inputs=dependencies):
            return tf.cond(pred=(temperature < epsilon), true_fn=fn_mode, false_fn=fn_sample)

    @tf_function(num_args=2)
    def log_probability(self, *, parameters, action):
        logits = parameters['logits']

        rank = self.action_spec.rank + 1
        action = tf.expand_dims(input=action, axis=rank)
        logit = tf.gather(params=logits, indices=action, batch_dims=rank)
        return tf.squeeze(input=logit, axis=rank)

    @tf_function(num_args=1)
    def entropy(self, *, parameters):
        probabilities, logits = parameters.get(('probabilities', 'logits'))

        return -tf.reduce_sum(input_tensor=(probabilities * logits), axis=-1)

    @tf_function(num_args=2)
    def kl_divergence(self, *, parameters1, parameters2):
        probabilities1, logits1 = parameters1.get(('probabilities', 'logits'))
        logits2 = parameters2['logits']

        log_prob_ratio = logits1 - logits2

        return tf.reduce_sum(input_tensor=(probabilities1 * log_prob_ratio), axis=-1)

    @tf_function(num_args=2)
    def action_value(self, *, parameters, action):
        action_values = parameters['action_values']

        rank = self.action_spec.rank + 1
        action = tf.expand_dims(input=action, axis=rank)
        action_value = tf.gather(params=action_values, indices=action, batch_dims=rank)

        return tf.squeeze(input=action_value, axis=rank)

    @tf_function(num_args=1)
    def state_value(self, *, parameters):
        return parameters['state_value']
