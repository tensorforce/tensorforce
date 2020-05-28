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
from tensorforce.core import layer_modules, Module
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):
    """
    Categorical distribution, for discrete integer actions (specification key: `categorical`).

    Args:
        name (string): Distribution name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        action_spec (specification): Action specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        embedding_shape (iter[int > 0]): Embedding shape
            (<span style="color:#0000C0"><b>internal use</b></span>).
        advantage_based (bool): Whether to compute action values as state value plus advantage
            (<span style="color:#00C000"><b>default</b></span>: false).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, action_spec, embedding_shape, advantage_based=False, summary_labels=None
    ):
        super().__init__(
            name=name, action_spec=action_spec, embedding_shape=embedding_shape,
            summary_labels=summary_labels
        )

        input_spec = dict(type='float', shape=self.embedding_shape)
        num_values = self.action_spec['num_values']

        self.state_value = None
        if len(self.embedding_shape) == 1:
            action_size = util.product(xs=self.action_spec['shape'])
            self.action_values = self.add_module(
                name='action_values', module='linear', modules=layer_modules,
                size=(action_size * num_values), input_spec=input_spec
            )
            if advantage_based:
                self.state_value = self.add_module(
                    name='states_value', module='linear', modules=layer_modules, size=action_size,
                    input_spec=input_spec
                )

        else:
            if advantage_based:
                raise TensorforceError.invalid(
                    name=name, argument='advantage_based', condition='embedding shape'
                )
            if len(self.embedding_shape) < 1 or len(self.embedding_shape) > 3:
                raise TensorforceError.value(
                    name=name, argument='embedding_shape', value=self.embedding_shape,
                    hint='invalid rank'
                )
            if self.embedding_shape[:-1] == self.action_spec['shape'][:-1]:
                size = self.action_spec['shape'][-1]
            elif self.embedding_shape[:-1] == self.action_spec['shape']:
                size = 1
            else:
                raise TensorforceError.value(
                    name=name, argument='embedding_shape', value=self.embedding_shape,
                    hint='not flattened and incompatible with action shape'
                )
            self.action_values = self.add_module(
                name='action_values', module='linear', modules=layer_modules,
                size=(size * num_values), input_spec=input_spec
            )

        Module.register_tensor(
            name=(self.name + '-probabilities'),
            spec=dict(type='float', shape=(self.action_spec['shape'] + (num_values,))),
            batched=True
        )
        Module.register_tensor(
            name=(self.name + '-values'),
            spec=dict(type='float', shape=(self.action_spec['shape'] + (num_values,))),
            batched=True
        )

    def tf_parametrize(self, x, mask):
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))
        shape = (-1,) + self.action_spec['shape'] + (self.action_spec['num_values'],)

        # Action values
        action_values = self.action_values.apply(x=x)
        action_values = tf.reshape(tensor=action_values, shape=shape)

        if self.state_value is None:
            # Implicit states value (TODO: experimental)
            states_value = tf.reduce_logsumexp(input_tensor=action_values, axis=-1)

        else:
            # Explicit states value and advantage-based action values
            states_value = self.state_value.apply(x=x)
            states_value = tf.reshape(tensor=states_value, shape=shape[:-1])
            action_values = tf.expand_dims(input=states_value, axis=-1) + action_values
            action_values -= tf.math.reduce_mean(input_tensor=action_values, axis=-1, keepdims=True)

        # TODO: before or after states_value?
        min_float = tf.fill(
            dims=tf.shape(input=action_values), value=util.tf_dtype(dtype='float').min
        )
        action_values = tf.where(condition=mask, x=action_values, y=min_float)

        # Softmax for corresponding probabilities
        probabilities = tf.nn.softmax(logits=action_values, axis=-1)

        # "Normalized" logits
        logits = tf.math.log(x=tf.maximum(x=probabilities, y=epsilon))

        Module.update_tensor(name=(self.name + '-probabilities'), tensor=probabilities)
        Module.update_tensor(name=(self.name + '-values'), tensor=action_values)

        return logits, probabilities, states_value, action_values

    def tf_sample(self, parameters, temperature):
        logits, probabilities, _, action_values = parameters

        summary_probs = probabilities
        for _ in range(len(self.action_spec['shape'])):
            summary_probs = tf.math.reduce_mean(input_tensor=summary_probs, axis=1)

        logits, probabilities = self.add_summary(
            label=('distributions', 'categorical'), name='probabilities', tensor=summary_probs,
            pass_tensors=(logits, probabilities), enumerate_last_rank=True
        )

        one = tf.constant(value=1.0, dtype=util.tf_dtype(dtype='float'))
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))

        # Deterministic: maximum likelihood action
        definite = tf.argmax(input=action_values, axis=-1)
        definite = tf.dtypes.cast(x=definite, dtype=util.tf_dtype('int'))

        # Set logits to minimal value
        min_float = tf.fill(dims=tf.shape(input=logits), value=util.tf_dtype(dtype='float').min)
        logits = logits / temperature
        logits = tf.where(condition=(probabilities < epsilon), x=min_float, y=logits)

        # Non-deterministic: sample action using Gumbel distribution
        uniform_distribution = tf.random.uniform(
            shape=tf.shape(input=logits), minval=epsilon, maxval=(one - epsilon),
            dtype=util.tf_dtype(dtype='float')
        )
        gumbel_distribution = -tf.math.log(x=-tf.math.log(x=uniform_distribution))
        sampled = tf.argmax(input=(logits + gumbel_distribution), axis=-1)
        sampled = tf.dtypes.cast(x=sampled, dtype=util.tf_dtype('int'))

        return tf.where(condition=(temperature < epsilon), x=definite, y=sampled)

    def tf_log_probability(self, parameters, action):
        logits, _, _, _ = parameters

        if util.tf_dtype(dtype='int') not in (tf.int32, tf.int64):
            action = tf.dtypes.cast(x=action, dtype=tf.int32)

        logits = tf.gather(
            params=logits, indices=tf.expand_dims(input=action, axis=-1), batch_dims=-1
        )

        return tf.squeeze(input=logits, axis=-1)

    def tf_entropy(self, parameters):
        logits, probabilities, _, _ = parameters

        return -tf.reduce_sum(input_tensor=(probabilities * logits), axis=-1)

    def tf_kl_divergence(self, parameters1, parameters2):
        logits1, probabilities1, _, _ = parameters1
        logits2, _, _, _ = parameters2

        log_prob_ratio = logits1 - logits2

        return tf.reduce_sum(input_tensor=(probabilities1 * log_prob_ratio), axis=-1)

    def tf_action_value(self, parameters, action=None):
        _, _, _, action_values = parameters

        if action is not None:
            if util.tf_dtype(dtype='int') not in (tf.int32, tf.int64):
                action = tf.dtypes.cast(x=action, dtype=tf.int32)

            action = tf.expand_dims(input=action, axis=-1)
            action_values = tf.gather(params=action_values, indices=action, batch_dims=-1)
            action_values = tf.squeeze(input=action_values, axis=-1)

        return action_values  # states_value + tf.squeeze(input=logits, axis=-1)

    def tf_states_value(self, parameters):
        _, _, states_value, _ = parameters

        return states_value
