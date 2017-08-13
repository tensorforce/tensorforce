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
"""
Beta distribution.
"""
import tensorflow as tf

from tensorforce import util
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


# TODO Michael: integrate to model, rescale min max
class Beta(Distribution):
    def __init__(self, shape, min_value, max_value, alpha, beta):
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
        self.shape = shape
        self.min_value = min_value
        self.h = (max_value - min_value) / 2
        self.alpha = alpha
        self.beta = beta

    def kl_divergence(self, other):
        assert isinstance(other, Beta)

        return other.log_norm - self.log_norm - tf.digamma(other.beta) * (other.beta - self.beta) - \
            tf.digamma(other.alpha) * (other.alpha - self.alpha) + tf.digamma(other.sum) * (other.sum - self.sum)

    def entropy(self):
        return self.log_norm - (self.beta - 1.0) * tf.digamma(self.beta) - \
               (self.alpha - 1.0) * tf.digamma(self.alpha) + ((self.sum - 2.0) * tf.digamma(self.sum))

    @classmethod
    def from_tensors(cls, parameters, deterministic):
        self = cls(shape=None, min_value=None, max_value=None)
        self.distribution = (self.alpha, self.beta) = parameters
        self.deterministic = deterministic

        return self

    def create_tf_operations(self, x, deterministic):
        # Flat mean and log standard deviation
        flat_size = util.prod(self.shape)
        self.alpha = layers['dense'](x=x, size=flat_size, bias=self.alpha, activation='softplus')
        self.beta = layers['dense'](x=x, size=flat_size, bias=self.beta, activation='softplus')

        self.sum = self.alpha + self.beta
        self.mean = self.alpha / self.sum

        self.log_norm = tf.lgamma(self.alpha) + tf.lgamma(self.beta) - tf.lgamma(self.sum)

        self.distribution = (self.alpha, self.beta)
        self.deterministic = deterministic

    def log_probability(self, action):
        return (self.alpha - 1.0) * tf.log(action) + (self.alpha - 1.0) * tf.log1p(-action) - self.log_norm

    def sample(self):
        deterministic = self.mean

        alpha_sample = tf.random_gamma(shape=tf.shape(input=self.alpha), alpha=self.alpha)
        beta_sample = tf.random_gamma(shape=tf.shape(input=self.beta), alpha=self.beta)
        sample = alpha_sample / (alpha_sample + beta_sample)

        return self.min_value + tf.where(condition=self.deterministic, x=deterministic, y=sample)
