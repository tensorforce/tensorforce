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

from math import log

import tensorflow as tf

from tensorforce import util
from tensorforce.core.networks import Linear
from tensorforce.core.distributions import Distribution


class Beta(Distribution):
    """
    Beta distribution, for bounded continuous actions
    """

    def __init__(self, shape, min_value, max_value, alpha=0.0, beta=0.0, scope='beta', summary_labels=()):
        """
        Beta distribution used for continuous actions. In particular, the Beta distribution
        allows to bound action values with min and max values.

        Args:
            shape: Shape of actions
            min_value: Min value of all actions for the given shape
            max_value: Max value of all actions for the given shape
            alpha: Concentration parameter of the Beta distribution
            beta: Concentration parameter of the Beta distribution
        """
        assert min_value is None or max_value > min_value
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        action_size = util.prod(self.shape)

        self.alpha = Linear(size=action_size, bias=alpha, scope='alpha')
        self.beta = Linear(size=action_size, bias=beta, scope='beta')

        super(Beta, self).__init__(scope, summary_labels)

    def tf_parameterize(self, x):
        # Softplus to ensure alpha and beta >= 1
        # epsilon < 1.0, hence negative
        log_eps = log(util.epsilon)

        alpha = self.alpha.apply(x=x)
        alpha = tf.clip_by_value(t=alpha, clip_value_min=log_eps, clip_value_max=-log_eps)
        alpha = tf.log(x=(tf.exp(x=alpha) + 1.0)) + 1.0

        beta = self.beta.apply(x=x)
        beta = tf.clip_by_value(t=beta, clip_value_min=log_eps, clip_value_max=-log_eps)
        beta = tf.log(x=(tf.exp(x=beta) + 1.0)) + 1.0

        shape = (-1,) + self.shape
        alpha = tf.reshape(tensor=alpha, shape=shape)
        beta = tf.reshape(tensor=beta, shape=shape)

        alpha_beta = tf.maximum(x=(alpha + beta), y=util.epsilon)
        log_norm = tf.lgamma(x=alpha) + tf.lgamma(x=beta) - tf.lgamma(x=alpha_beta)

        return alpha, beta, alpha_beta, log_norm

    def tf_sample(self, distr_params, deterministic):
        alpha, beta, alpha_beta, _ = distr_params

        # Deterministic: mean as action
        definite = beta / alpha_beta

        # Non-deterministic: sample action using gamma distribution
        alpha_sample = tf.random_gamma(shape=(), alpha=alpha)
        beta_sample = tf.random_gamma(shape=(), alpha=beta)

        sampled = beta_sample / tf.maximum(x=(alpha_sample + beta_sample), y=util.epsilon)

        return self.min_value + (self.max_value - self.min_value) * \
            tf.where(condition=deterministic, x=definite, y=sampled)

    def tf_log_probability(self, distr_params, action):
        alpha, beta, _, log_norm = distr_params
        action = (action - self.min_value) / (self.max_value - self.min_value)
        action = tf.minimum(x=action, y=(1.0 - util.epsilon))
        return (beta - 1.0) * tf.log(x=tf.maximum(x=action, y=util.epsilon)) + \
            (alpha - 1.0) * tf.log1p(x=-action) - log_norm

    def tf_entropy(self, distr_params):
        alpha, beta, alpha_beta, log_norm = distr_params
        return log_norm - (beta - 1.0) * tf.digamma(x=beta) - (alpha - 1.0) * tf.digamma(x=alpha) + \
            (alpha_beta - 2.0) * tf.digamma(x=alpha_beta)

    def tf_kl_divergence(self, distr_params1, distr_params2):
        alpha1, beta1, alpha_beta1, log_norm1 = distr_params1
        alpha2, beta2, alpha_beta2, log_norm2 = distr_params2
        return log_norm2 - log_norm1 - tf.digamma(x=beta1) * (beta2 - beta1) - \
            tf.digamma(x=alpha1) * (alpha2 - alpha1) + tf.digamma(x=alpha_beta1) * (alpha_beta2 - alpha_beta1)

    def tf_regularization_loss(self):
        regularization_loss = super(Beta, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.alpha.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.beta.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        distribution_variables = super(Beta, self).get_variables(include_non_trainable=include_non_trainable)
        alpha_variables = self.alpha.get_variables(include_non_trainable=include_non_trainable)
        beta_variables = self.beta.get_variables(include_non_trainable=include_non_trainable)

        return distribution_variables + alpha_variables + beta_variables

    def get_summaries(self):
        distribution_summaries = super(Beta, self).get_summaries()
        alpha_summaries = self.alpha.get_summaries()
        beta_summaries = self.beta.get_summaries()

        return distribution_summaries + alpha_summaries + beta_summaries
