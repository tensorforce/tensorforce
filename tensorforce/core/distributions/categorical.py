# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from math import log
import tensorflow as tf

from tensorforce import util
from tensorforce.core.networks import Linear
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):
    """
    Categorical distribution, for discrete actions.
    """

    def __init__(self, shape, num_actions, probabilities=None, scope='categorical', summary_labels=()):
        """
        Categorical distribution.

        Args:
            shape: Action shape.
            num_actions: Number of discrete action alternatives.
            probabilities: Optional distribution bias.
        """
        self.num_actions = num_actions

        action_size = util.prod(shape) * self.num_actions
        if probabilities is None:
            logits = 0.0
        else:
            logits = [log(prob) for _ in range(util.prod(shape)) for prob in probabilities]
        self.logits = Linear(size=action_size, bias=logits, scope='logits', summary_labels=summary_labels)

        super(Categorical, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_parameterize(self, x):
        # Flat logits
        logits = self.logits.apply(x=x)

        # Reshape logits to action shape
        shape = (-1,) + self.shape + (self.num_actions,)
        logits = tf.reshape(tensor=logits, shape=shape)

        # !!!
        state_value = tf.reduce_logsumexp(input_tensor=logits, axis=-1)

        # Softmax for corresponding probabilities
        probabilities = tf.nn.softmax(logits=logits, axis=-1)

        # Min epsilon probability for numerical stability
        probabilities = tf.maximum(x=probabilities, y=util.epsilon)

        # "Normalized" logits
        logits = tf.log(x=probabilities)

        if 'distribution' in self.summary_labels:
            for n in range(self.num_actions):
                tf.contrib.summary.scalar(
                    name=(self.scope + '-action' + str(n)),
                    tensor=probabilities[:, n]
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
            one_hot = tf.one_hot(indices=action, depth=self.num_actions)
            logits = tf.reduce_sum(input_tensor=(logits * one_hot), axis=-1)
        return state_value + logits

    def tf_sample(self, distr_params, deterministic):
        logits, _, _ = distr_params

        # Deterministic: maximum likelihood action
        definite = tf.argmax(input=logits, axis=-1, output_type=util.tf_dtype('int'))

        # Non-deterministic: sample action using Gumbel distribution
        uniform_distribution = tf.random_uniform(
            shape=tf.shape(input=logits),
            minval=util.epsilon,
            maxval=(1.0 - util.epsilon)
        )
        gumbel_distribution = -tf.log(x=-tf.log(x=uniform_distribution))
        sampled = tf.argmax(input=(logits + gumbel_distribution), axis=-1, output_type=util.tf_dtype('int'))

        return tf.where(condition=deterministic, x=definite, y=sampled)

    def tf_log_probability(self, distr_params, action):
        logits, _, _ = distr_params
        one_hot = tf.one_hot(indices=action, depth=self.num_actions)
        return tf.reduce_sum(input_tensor=(logits * one_hot), axis=-1)

    def tf_entropy(self, distr_params):
        logits, probabilities, _ = distr_params
        return -tf.reduce_sum(input_tensor=(probabilities * logits), axis=-1)

    def tf_kl_divergence(self, distr_params1, distr_params2):
        logits1, probabilities1, _ = distr_params1
        logits2, _, _ = distr_params2
        log_prob_ratio = logits1 - logits2
        return tf.reduce_sum(input_tensor=(probabilities1 * log_prob_ratio), axis=-1)

    def tf_regularization_loss(self):
        regularization_loss = super(Categorical, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.logits.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        distribution_variables = super(Categorical, self).get_variables(include_nontrainable=include_nontrainable)
        logits_variables = self.logits.get_variables(include_nontrainable=include_nontrainable)

        return distribution_variables + logits_variables
