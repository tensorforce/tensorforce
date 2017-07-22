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
Standard Gaussian policy.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorforce import util, TensorForceError
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Gaussian(Distribution):

    def __init__(self, mean=0.0, log_stddev=0.0, min_value=None, max_value=None):
        if min_value is not None or max_value is not None:
            raise TensorForceError('Min/max value not allowed for Gaussian.')
        self.mean = mean
        self.log_stddev = log_stddev

    @classmethod
    def from_tensors(cls, parameters):
        self = cls()
        self.distribution = (self.mean, self.log_stddev) = parameters
        return self

    def create_tf_operations(self, x, deterministic):
        self.mean = layers['linear'](x=x, size=1, bias=(self.mean,))
        self.mean = tf.squeeze(input=self.mean, axis=1)
        self.log_stddev = layers['linear'](x=x, size=1, bias=(self.log_stddev,))
        self.log_stddev = tf.squeeze(input=self.log_stddev, axis=1)
        self.log_stddev = tf.minimum(x=self.log_stddev, y=10.0)  # prevent infinity when exp
        self.distribution = (self.mean, self.log_stddev)
        self.deterministic = deterministic

    def sample(self):
        deterministic = self.mean
        stddev = tf.exp(x=self.log_stddev)
        sampled = self.mean + stddev * tf.random_normal(shape=tf.shape(self.mean))
        return tf.where(condition=self.deterministic, x=deterministic, y=sampled)

    def log_probability(self, action):
        l2_dist = tf.square(action - self.mean)
        sqr_stddev = tf.square(x=tf.exp(x=self.log_stddev))
        log_prob = -l2_dist / (2 * sqr_stddev + util.epsilon) - 0.5 * tf.log(tf.constant(2 * np.pi)) - self.log_stddev
        return log_prob

    def entropy(self):
        entropy = tf.reduce_mean(self.log_stddev + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32), axis=0)
        return entropy

    def kl_divergence(self, other):
        assert isinstance(other, Gaussian)
        l2_dist = tf.square(self.mean - other.mean)
        stddev1 = tf.exp(x=self.log_stddev)
        sqr_stddev2 = tf.square(x=tf.exp(x=other.log_stddev))
        kl_div = tf.reduce_mean(self.log_stddev - other.log_stddev + (stddev1 + l2_dist) / (2 * sqr_stddev2 + util.epsilon) - 0.5, axis=0)
        return kl_div
