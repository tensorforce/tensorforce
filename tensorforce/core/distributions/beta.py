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
from math import log

import tensorflow as tf

from tensorforce import util
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Beta(Distribution):

    def __init__(self, min_value, max_value, shape, alpha=0.0, beta=0.0):
        """
        Beta distribution used for continuous actions. In particular, the Beta distribution
        allows to bound action values with min and max values.

        Args:
            min_value: Min value of all actions for the given shape
            max_value: Max value of all actions for the given shape
            shape: Shape of actions
            alpha: Concentration parameter of the Beta distribution
            beta: Concentration parameter of the Beta distribution
        """
        assert min_value is None or max_value > min_value
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        self.alpha = alpha
        self.beta = beta

    @classmethod
    def from_tensors(cls, tensors, deterministic):
        self = cls(min_value=None, max_value=None, shape=None)
        self.min_value, self.max_value, self.alpha, self.beta = tensors
        self.sum = self.alpha + self.beta
        self.mean = self.beta / tf.maximum(x=self.sum, y=util.epsilon)
        self.log_norm = tf.lgamma(self.alpha) + tf.lgamma(self.beta) - tf.lgamma(self.sum)
        self.deterministic = deterministic
        return self

    def get_tensors(self):
        return (self.min_value, self.max_value, self.alpha, self.beta)

    def create_tf_operations(self, x, deterministic):
        self.min_value = tf.constant(value=self.min_value, dtype=tf.float32)
        self.max_value = tf.constant(value=self.max_value, dtype=tf.float32)

        # Flat mean and log standard deviation
        flat_size = util.prod(self.shape)
        log_eps = log(util.epsilon)

        # Softplus to ensure alpha and beta >= 1
        self.alpha = layers['linear'](x=x, size=flat_size, bias=self.alpha)
        self.alpha = tf.clip_by_value(t=self.alpha, clip_value_min=log_eps, clip_value_max=-log_eps)
        self.alpha = tf.log(x=(tf.exp(x=self.alpha) + 1.0))  # tf.nn.softplus(features=self.alpha)

        self.beta = layers['linear'](x=x, size=flat_size, bias=self.beta)
        self.beta = tf.clip_by_value(t=self.beta, clip_value_min=log_eps, clip_value_max=-log_eps)
        self.beta = tf.log(x=(tf.exp(x=self.beta) + 1.0))  # tf.nn.softplus(features=self.beta)

        shape = (-1,) + self.shape
        self.alpha = tf.reshape(tensor=self.alpha, shape=shape)
        self.beta = tf.reshape(tensor=self.beta, shape=shape)

        self.sum = tf.maximum(x=(self.alpha + self.beta), y=util.epsilon)
        self.mean = self.beta / self.sum

        self.log_norm = tf.lgamma(self.alpha) + tf.lgamma(self.beta) - tf.lgamma(self.sum)

        self.deterministic = deterministic

    def log_probability(self, action):
        action = (action - self.min_value) / (self.max_value - self.min_value)
        action = tf.minimum(x=action, y=(1.0 - util.epsilon))

        return (self.beta - 1.0) * tf.log(tf.maximum(action, util.epsilon)) + (self.alpha - 1.0) * tf.log1p(-action) - self.log_norm

    def kl_divergence(self, other):
        assert isinstance(other, Beta)

        return other.log_norm - self.log_norm - tf.digamma(self.beta) * (other.beta - self.beta) - tf.digamma(self.alpha) * (other.alpha - self.alpha) + tf.digamma(self.sum) * (other.sum - self.sum)

    def entropy(self):
        return self.log_norm - (self.beta - 1.0) * tf.digamma(self.beta) - (self.alpha - 1.0) * tf.digamma(self.alpha) + (self.sum - 2.0) * tf.digamma(self.sum)

    def sample(self):
        deterministic = self.mean

        alpha_sample = tf.random_gamma(shape=(), alpha=self.alpha)
        beta_sample = tf.random_gamma(shape=(), alpha=self.beta)

        sample = beta_sample / tf.maximum(x=(alpha_sample + beta_sample), y=util.epsilon)

        return self.min_value + tf.where(condition=self.deterministic, x=deterministic, y=sample) * (self.max_value - self.min_value)
