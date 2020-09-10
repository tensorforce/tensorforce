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

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules, SignatureDict, TensorDict, TensorSpec, TensorsSpec, \
    tf_function, tf_util
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):
    """
    Categorical distribution, for discrete integer actions (specification key: `categorical`).

    Args:
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, name=None, action_spec=None, input_spec=None):
        assert action_spec.type == 'int' and action_spec.num_values is not None

        parameters_spec = TensorsSpec(
            logits=TensorSpec(type='float', shape=(action_spec.shape + (action_spec.num_values,))),
            probabilities=TensorSpec(
                type='float', shape=(action_spec.shape + (action_spec.num_values,))
            ),
            state_value=TensorSpec(type='float', shape=action_spec.shape),
            action_values=TensorSpec(
                type='float', shape=(action_spec.shape + (action_spec.num_values,))
            )
        )
        conditions_spec = TensorsSpec()

        super().__init__(
            name=name, action_spec=action_spec, input_spec=input_spec,
            parameters_spec=parameters_spec, conditions_spec=conditions_spec
        )

        if self.config.enable_int_action_masking:
            self.conditions_spec['mask'] = TensorSpec(
                type='bool', shape=(self.action_spec.shape + (self.action_spec.num_values,))
            )

        num_values = self.action_spec.num_values
        if len(self.input_spec.shape) == 1:
            # Single embedding
            self.action_values = self.submodule(
                name='action_values', module='linear', modules=layer_modules,
                size=(self.action_spec.size * num_values), initialization_scale=0.01,
                input_spec=input_spec
            )

        else:
            # Embedding per action
            if len(self.input_spec.shape) < 1 or len(self.input_spec.shape) > 3:
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

    def initialize(self):
        super().initialize()

        prefix = 'distributions/' + self.name + '-probability'
        names = [prefix + str(n) for n in range(self.action_spec.num_values)]
        self.register_summary(label='distribution', name=names)

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        shape = (-1,) + self.action_spec.shape + (self.action_spec.num_values,)

        # Action values
        action_values = self.action_values.apply(x=x)
        action_values = tf.reshape(tensor=action_values, shape=shape)

        # Implicit states value (TODO: experimental)
        state_value = tf.reduce_logsumexp(input_tensor=action_values, axis=-1)

        # # Explicit states value and advantage-based action values
        # state_value = self.state_value.apply(x=x)
        # state_value = tf.reshape(tensor=state_value, shape=shape[:-1])
        # action_values = tf.expand_dims(input=state_value, axis=-1) + action_values
        # action_values -= tf.math.reduce_mean(input_tensor=action_values, axis=-1, keepdims=True)

        # Masking (TODO: before or after above?)
        if self.config.enable_int_action_masking:
            min_float = tf.fill(
                dims=tf.shape(input=action_values), value=tf_util.get_dtype(type='float').min
            )
            action_values = tf.where(condition=conditions['mask'], x=action_values, y=min_float)

        # Softmax for corresponding probabilities
        probabilities = tf.nn.softmax(logits=action_values, axis=-1)

        # "Normalized" logits
        # logits = tf.math.log(x=tf.maximum(x=probabilities, y=epsilon))
        # logits = tf.nn.log_softmax(logits=action_values, axis=-1)
        logits = action_values - tf.expand_dims(input=state_value, axis=-1)

        return TensorDict(
            logits=logits, probabilities=probabilities, state_value=state_value,
            action_values=action_values
        )

    @tf_function(num_args=1)
    def mode(self, *, parameters):
        action_values = parameters['action_values']

        action = tf.math.argmax(input=action_values, axis=-1)
        return tf_util.cast(x=action, dtype='int')

    @tf_function(num_args=2)
    def sample(self, *, parameters, temperature):
        logits, probabilities, action_values = parameters.get(
            ('logits', 'probabilities', 'action_values')
        )

        # Distribution parameter summaries
        def fn_summary():
            axis = range(self.action_spec.rank + 1)
            probs = tf.math.reduce_mean(input_tensor=probabilities, axis=axis)
            return [probs[n] for n in range(self.action_spec.num_values)]

        prefix = 'distributions/' + self.name + '-probability'
        names = [prefix + str(n) for n in range(self.action_spec.num_values)]
        dependencies = self.summary(
            label='distribution', name=names, data=fn_summary, step='timesteps'
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
        logits, probabilities = parameters.get(('logits', 'probabilities'))

        return -tf.reduce_sum(input_tensor=(probabilities * logits), axis=-1)

    @tf_function(num_args=2)
    def kl_divergence(self, *, parameters1, parameters2):
        logits1, probabilities1 = parameters1.get(('logits', 'probabilities'))
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
        # TODO: state_value + tf.squeeze(input=logits, axis=-1)

    @tf_function(num_args=1)
    def state_value(self, *, parameters):
        return parameters['state_value']
