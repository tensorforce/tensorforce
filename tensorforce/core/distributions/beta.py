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
Beta distribution for bounded continuous action spaces.
"""
import tensorflow as tf

from tensorforce import util
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Beta(Distribution):
    def __init__(self, shape, min_value, max_value, alpha=0, beta=0):
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
        assert max_value > min_value

        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
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
    def from_tensors(cls, tensors, deterministic):
        self = cls(shape=None, min_value=None, max_value=None)
        self.alpha, self.beta = tensors
        self.deterministic = deterministic

        return self

    def get_tensors(self):
        return (self.alpha, self.beta)

    def create_tf_operations(self, x, deterministic):
        # Flat mean and log standard deviation
        flat_size = util.prod(self.shape)

        # Softplus to ensure alpha and beta >= 1
        self.alpha = layers['linear'](x=x, size=flat_size, bias=self.alpha)
        self.alpha = tf.nn.softplus(features=self.alpha)

        self.beta = layers['linear'](x=x, size=flat_size, bias=self.beta)
        self.beta = tf.nn.softplus(features=self.beta)

        self.sum = self.alpha + self.beta
        self.mean = self.alpha / self.sum

        self.log_norm = tf.lgamma(self.alpha) + tf.lgamma(self.beta) - tf.lgamma(self.sum)

        self.deterministic = deterministic

    def log_probability(self, action):
        return (self.alpha - 1.0) * tf.log(action) + (self.alpha - 1.0) * tf.log1p(-action) - self.log_norm

    def sample(self):
        deterministic = self.mean
        print(tf.shape(deterministic))
        print(tf.shape(self.alpha))

        alpha_sample = tf.random_gamma(shape=tf.shape(input=self.alpha), alpha=self.alpha)
        beta_sample = tf.random_gamma(shape=tf.shape(input=self.beta), alpha=self.beta)
        sample = alpha_sample / (alpha_sample + beta_sample)

        print(tf.shape(alpha_sample))

        return self.min_value + tf.where(condition=self.deterministic, x=deterministic, y=sample) * \
                                (self.max_value - self.min_value)
