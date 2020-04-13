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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules, TensorDict, TensorSpec, TensorsSpec, tf_function, \
    tf_util
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):
    """
    Categorical distribution, for discrete integer actions (specification key: `categorical`).

    Args:
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, summary_labels=None, name=None, action_spec=None, input_spec=None):
        parameters_spec = TensorsSpec(
            logits=TensorSpec(type='float', shape=(action_spec.shape + (action_spec.num_values,))),
            probabilities=TensorSpec(
                type='float', shape=(action_spec.shape + (action_spec.num_values,))
            ),
            states_value=TensorSpec(type='float', shape=action_spec.shape),
            action_values=TensorSpec(
                type='float', shape=(action_spec.shape + (action_spec.num_values,))
            )
        )

        super().__init__(
            summary_labels=summary_labels, name=name, action_spec=action_spec,
            input_spec=input_spec, parameters_spec=parameters_spec
        )
        num_values = self.action_spec.num_values

        if len(self.input_spec.shape) == 1:
            action_size = util.product(xs=self.action_spec.shape)
            self.deviations = self.add_module(
                name='deviations', module='linear', modules=layer_modules,
                size=(action_size * num_values), input_spec=self.input_spec
            )

        else:
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
            self.deviations = self.add_module(
                name='deviations', module='linear', modules=layer_modules,
                size=(size * num_values), input_spec=self.input_spec
            )

    def input_signature(self, function):
        if function == 'parametrize':
            return [
                self.input_spec.signature(batched=True),
                TensorSpec(
                    type='bool', shape=(self.action_spec.shape + (self.action_spec.num_values,))
                ).signature(batched=True)
            ]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=2)
    def parametrize(self, x, mask):
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        shape = (-1,) + self.action_spec.shape + (self.action_spec.num_values,)

        # Deviations
        action_values = self.deviations.apply(x=x)
        action_values = tf.reshape(tensor=action_values, shape=shape)
        min_float = tf.fill(
            dims=tf.shape(input=action_values), value=tf_util.get_dtype(type='float').min
        )

        # States value
        action_values = tf.where(condition=mask, x=action_values, y=min_float)
        states_value = tf.reduce_logsumexp(input_tensor=action_values, axis=-1)

        # Softmax for corresponding probabilities
        probabilities = tf.nn.softmax(logits=action_values, axis=-1)

        # "Normalized" logits
        logits = tf.math.log(x=tf.maximum(x=probabilities, y=epsilon))

        return TensorDict(
            logits=logits, probabilities=probabilities, states_value=states_value,
            action_values=action_values
        )

    @tf_function(num_args=2)
    def sample(self, parameters, temperature):
        logits, probabilities = parameters.get('logits', 'probabilities')

        summary_probs = probabilities
        for _ in range(len(self.action_spec.shape)):
            summary_probs = tf.math.reduce_mean(input_tensor=summary_probs, axis=1)

        logits, probabilities = self.add_summary(
            label=('distributions', 'categorical'), name='probabilities', tensor=summary_probs,
            pass_tensors=(logits, probabilities), enumerate_last_rank=True
        )

        one = tf_util.constant(value=1.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        # Deterministic: maximum likelihood action
        definite = tf.argmax(input=logits, axis=-1)
        definite = tf_util.cast(x=definite, dtype='int')

        # Set logits to minimal value
        min_float = tf.fill(dims=tf.shape(input=logits), value=tf_util.get_dtype(type='float').min)
        logits = logits / temperature
        logits = tf.where(condition=(probabilities < epsilon), x=min_float, y=logits)

        # Non-deterministic: sample action using Gumbel distribution
        uniform_distribution = tf.random.uniform(
            shape=tf.shape(input=logits), minval=epsilon, maxval=(one - epsilon),
            dtype=tf_util.get_dtype(type='float')
        )
        gumbel_distribution = -tf.math.log(x=-tf.math.log(x=uniform_distribution))
        sampled = tf.argmax(input=(logits + gumbel_distribution), axis=-1)
        sampled = tf_util.cast(x=sampled, dtype='int')

        return tf.where(condition=(temperature < epsilon), x=definite, y=sampled)

    @tf_function(num_args=2)
    def log_probability(self, parameters, action):
        logits = parameters['logits']

        action = tf.expand_dims(input=tf_util.int32(x=action), axis=-1)
        logits = tf.gather(params=logits, indices=action, batch_dims=-1)

        return tf.squeeze(input=logits, axis=-1)

    @tf_function(num_args=1)
    def entropy(self, parameters):
        logits, probabilities = parameters.get('logits', 'probabilities')

        return -tf.reduce_sum(input_tensor=(probabilities * logits), axis=-1)

    @tf_function(num_args=2)
    def kl_divergence(self, parameters1, parameters2):
        logits1, probabilities1 = parameters1.get('logits', 'probabilities')
        logits2 = parameters2['logits']

        log_prob_ratio = logits1 - logits2

        return tf.reduce_sum(input_tensor=(probabilities1 * log_prob_ratio), axis=-1)

    @tf_function(num_args=2)
    def action_value(self, parameters, action):
        action_values = parameters['action_values']

        action = tf.expand_dims(input=tf_util.int32(x=action), axis=-1)
        action_values = tf.gather(params=action_values, indices=action, batch_dims=-1)
        action_values = tf.squeeze(input=action_values, axis=-1)

        return action_values  # TODO: states_value + tf.squeeze(input=logits, axis=-1)

    @tf_function(num_args=1)
    def states_value(self, parameters):
        states_value = parameters['states_value']

        return states_value
