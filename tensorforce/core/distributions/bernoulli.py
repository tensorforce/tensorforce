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


class Bernoulli(Distribution):
    """
    Bernoulli distribution, for binary boolean actions.
    """

    def __init__(self, shape, probability=0.5, scope='bernoulli', summary_labels=()):
        """
        Bernoulli distribution.

        Args:
            shape: Action shape.
            probability: Optional distribution bias.
        """
        self.shape = shape
        action_size = util.prod(self.shape)

        self.logit = Linear(size=action_size, bias=log(probability), scope='logit', summary_labels=summary_labels)

        super(Bernoulli, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_parameterize(self, x):
        # Flat logit
        logit = self.logit.apply(x=x)

        # Reshape logit to action shape
        shape = (-1,) + self.shape
        logit = tf.reshape(tensor=logit, shape=shape)

        # TODO rename
        state_value = logit

        # Sigmoid for corresponding probability
        probability = tf.sigmoid(x=logit)

        # Min epsilon probability for numerical stability
        probability = tf.clip_by_value(
            t=probability,
            clip_value_min=util.epsilon,
            clip_value_max=(1.0 - util.epsilon)
        )

        # "Normalized" logits
        true_logit = tf.log(x=probability)
        false_logit = tf.log(x=(1.0 - probability))

        if 'distribution' in self.summary_labels:
            tf.contrib.summary.scalar(name=self.scope, tensor=probability)

        return true_logit, false_logit, probability, state_value

    def state_value(self, distr_params):
        _, _, _, state_value = distr_params
        return state_value

    def state_action_value(self, distr_params, action=None):
        true_logit, false_logit, _, state_value = distr_params
        if action is None:
            state_value = tf.expand_dims(input=state_value, axis=-1)
            logits = tf.stack(values=(false_logit, true_logit), axis=-1)
        else:
            logits = tf.where(condition=action, x=true_logit, y=false_logit)
        return state_value + logits

    def tf_sample(self, distr_params, deterministic):
        _, _, probability, _ = distr_params

        # Deterministic: true if >= 0.5
        definite = tf.greater_equal(x=probability, y=0.5)

        # Non-deterministic: sample true if >= uniform distribution
        uniform = tf.random_uniform(shape=tf.shape(probability))
        sampled = tf.greater_equal(x=probability, y=uniform)

        return tf.where(condition=deterministic, x=definite, y=sampled)

    def tf_log_probability(self, distr_params, action):
        true_logit, false_logit, _, _ = distr_params
        return tf.where(condition=action, x=true_logit, y=false_logit)

    def tf_entropy(self, distr_params):
        true_logit, false_logit, probability, _ = distr_params
        return -probability * true_logit - (1.0 - probability) * false_logit

    def tf_kl_divergence(self, distr_params1, distr_params2):
        true_logit1, false_logit1, probability1, _ = distr_params1
        true_logit2, false_logit2, _, _ = distr_params2
        true_log_prob_ratio = true_logit1 - true_logit2
        false_log_prob_ratio = false_logit1 - false_logit2
        return probability1 * true_log_prob_ratio + (1.0 - probability1) * false_log_prob_ratio

    def tf_regularization_loss(self):
        regularization_loss = super(Bernoulli, self).tf_regularization_loss()
        if super(Bernoulli, self).tf_regularization_loss() is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.logit.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        distribution_variables = super(Bernoulli, self).get_variables(include_nontrainable=include_nontrainable)
        logit_variables = self.logit.get_variables(include_nontrainable=include_nontrainable)

        return distribution_variables + logit_variables
