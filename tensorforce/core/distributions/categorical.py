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

from tensorforce import util
from tensorforce.core import layer_modules
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):
    """
    Categorical distribution, for discrete integer actions (specification key: `categorical`).

    Args:
        name (string): Distribution name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        action_spec (specification): Action specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        embedding_size (int > 0): Embedding size
            (<span style="color:#0000C0"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, action_spec, embedding_size, summary_labels=None):
        super().__init__(
            name=name, action_spec=action_spec, embedding_size=embedding_size,
            summary_labels=summary_labels
        )

        action_size = util.product(xs=self.action_spec['shape']) * self.action_spec['num_values']
        input_spec = dict(type='float', shape=(self.embedding_size,))
        self.logits = self.add_module(
            name='logits', module='linear', modules=layer_modules, size=action_size,
            input_spec=input_spec
        )

    def tf_parametrize(self, x, mask):
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))
        shape = (-1,) + self.action_spec['shape'] + (self.action_spec['num_values'],)

        # Logits
        logits = self.logits.apply(x=x)
        logits = tf.reshape(tensor=logits, shape=shape)
        min_float = tf.fill(dims=tf.shape(input=logits), value=util.tf_dtype(dtype='float').min)
        logits = tf.where(condition=mask, x=logits, y=min_float)

        # States value
        states_value = tf.reduce_logsumexp(input_tensor=logits, axis=-1)

        # Softmax for corresponding probabilities
        probabilities = tf.nn.softmax(logits=logits, axis=-1)

        # "Normalized" logits
        logits = tf.log(x=tf.maximum(x=probabilities, y=epsilon))

        # Logits as pass_tensor since used for sampling
        logits, probabilities, states_value = self.add_summary(
            label=('distributions', 'categorical'), name='probability', tensor=probabilities,
            pass_tensors=(logits, probabilities, states_value), enumerate_last_rank=True
        )

        return logits, probabilities, states_value

    def tf_sample(self, parameters, deterministic):
        logits, probabilities, _ = parameters

        # Deterministic: maximum likelihood action
        definite = tf.argmax(input=logits, axis=-1)
        definite = tf.dtypes.cast(x=definite, dtype=util.tf_dtype('int'))

        one = tf.constant(value=1.0, dtype=util.tf_dtype(dtype='float'))
        epsilon = tf.constant(value=util.epsilon, dtype=util.tf_dtype(dtype='float'))

        # Set logits to minimal value
        min_float = tf.fill(dims=tf.shape(input=logits), value=util.tf_dtype(dtype='float').min)
        logits = tf.where(condition=(probabilities < epsilon), x=min_float, y=logits)

        # Non-deterministic: sample action using Gumbel distribution
        uniform_distribution = tf.random.uniform(
            shape=tf.shape(input=logits), minval=epsilon, maxval=(one - epsilon),
            dtype=util.tf_dtype(dtype='float')
        )
        gumbel_distribution = -tf.log(x=-tf.log(x=uniform_distribution))
        sampled = tf.argmax(input=(logits + gumbel_distribution), axis=-1)
        sampled = tf.dtypes.cast(x=sampled, dtype=util.tf_dtype('int'))

        return tf.where(condition=deterministic, x=definite, y=sampled)

    def tf_log_probability(self, parameters, action):
        logits, _, _ = parameters

        if util.tf_dtype(dtype='int') not in (tf.int32, tf.int64):
            action = tf.dtypes.cast(x=action, dtype=tf.int32)

        # better way?
        one_hot = tf.one_hot(
            indices=action, depth=self.action_spec['num_values'],
            dtype=util.tf_dtype(dtype='float')
        )

        return tf.reduce_sum(input_tensor=(logits * one_hot), axis=-1)

    def tf_entropy(self, parameters):
        logits, probabilities, _ = parameters

        return -tf.reduce_sum(input_tensor=(probabilities * logits), axis=-1)

    def tf_kl_divergence(self, parameters1, parameters2):
        logits1, probabilities1, _ = parameters1
        logits2, _, _ = parameters2

        log_prob_ratio = logits1 - logits2

        return tf.reduce_sum(input_tensor=(probabilities1 * log_prob_ratio), axis=-1)

    def tf_action_value(self, parameters, action=None):
        logits, _, states_value = parameters

        if action is None:
            states_value = tf.expand_dims(input=states_value, axis=-1)

        else:
            if util.tf_dtype(dtype='int') not in (tf.int32, tf.int64):
                action = tf.dtypes.cast(x=action, dtype=tf.int32)

            one_hot = tf.one_hot(
                indices=action, depth=self.action_spec['num_values'],
                dtype=util.tf_dtype(dtype='float')
            )
            logits = tf.reduce_sum(input_tensor=(logits * one_hot), axis=-1)

        return states_value + logits

    def tf_states_value(self, parameters):
        _, _, states_value = parameters

        return states_value
