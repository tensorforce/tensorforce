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
    Categorical distribution, for discrete actions.
    """

    def __init__(self, name, action_spec, embedding_size, summary_labels=None):
        """
        Categorical distribution.
        """
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

    def tf_parametrize(self, x):
        # Flat logits
        logits = self.logits.apply(x=x)

        # Reshape logits to action shape
        shape = (-1,) + self.action_spec['shape'] + (self.action_spec['num_values'],)
        logits = tf.reshape(tensor=logits, shape=shape)

        # !!!
        state_value = tf.reduce_logsumexp(input_tensor=logits, axis=-1)

        # Softmax for corresponding probabilities
        probabilities = tf.nn.softmax(logits=logits, axis=-1)

        # Min epsilon probability for numerical stability
        probabilities = tf.maximum(x=probabilities, y=util.epsilon)

        # "Normalized" logits
        logits = tf.log(x=probabilities)

        # Logits as pass_tensor since used for sampling
        logits, probabilities, state_value = self.add_summary(
            label=('distributions', 'categorical'), name='probability', tensor=probabilities,
            pass_tensors=(logits, probabilities, state_value), enumerate_last_rank=True
        )

        return logits, probabilities, state_value

    def state_value(self, distr_params):
        _, _, state_value = distr_params
        return state_value

    def state_action_value(self, distr_params, action=None):
        logits, _, state_value = distr_params
        if action is None:
            state_value = tf.expand_dims(input=state_value, axis=-1)
        else:
            one_hot = tf.one_hot(
                indices=tf.dtypes.cast(x=action, dtype=tf.int32),
                depth=self.action_spec['num_values'], dtype=util.tf_dtype(dtype='float')
            )
            logits = tf.reduce_sum(input_tensor=(logits * one_hot), axis=-1)
        return state_value + logits

    def tf_sample(self, distr_params, deterministic):
        logits, _, _ = distr_params

        # Deterministic: maximum likelihood action
        definite = tf.argmax(input=logits, axis=-1)
        definite = tf.dtypes.cast(x=definite, dtype=util.tf_dtype('int'))

        # Non-deterministic: sample action using Gumbel distribution
        uniform_distribution = tf.random.uniform(
            shape=tf.shape(input=logits), minval=util.epsilon, maxval=(1.0 - util.epsilon),
            dtype=util.tf_dtype(dtype='float')
        )
        gumbel_distribution = -tf.log(x=-tf.log(x=uniform_distribution))
        sampled = tf.argmax(input=(logits + gumbel_distribution), axis=-1)
        sampled = tf.dtypes.cast(x=sampled, dtype=util.tf_dtype('int'))

        return tf.where(condition=deterministic, x=definite, y=sampled)

    def tf_log_probability(self, distr_params, action):
        logits, _, _ = distr_params
        one_hot = tf.one_hot(
            indices=tf.dtypes.cast(x=action, dtype=tf.int32), depth=self.action_spec['num_values'],
            dtype=util.tf_dtype(dtype='float')
        )
        return tf.reduce_sum(input_tensor=(logits * one_hot), axis=-1)

    def tf_entropy(self, distr_params):
        logits, probabilities, _ = distr_params
        return -tf.reduce_sum(input_tensor=(probabilities * logits), axis=-1)

    def tf_kl_divergence(self, distr_params1, distr_params2):
        logits1, probabilities1, _ = distr_params1
        logits2, _, _ = distr_params2
        log_prob_ratio = logits1 - logits2
        return tf.reduce_sum(input_tensor=(probabilities1 * log_prob_ratio), axis=-1)
