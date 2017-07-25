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

from tensorforce import util
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Gaussian(Distribution):

    def __init__(self, shape, min_value, max_value, mean=0.0, log_stddev=0.0):
        # TODO: warning
        # if min_value is not None or max_value is not None:
        #     raise TensorForceError('Min/max value not allowed for Gaussian.')
        self.shape = shape
        self.mean = mean
        self.log_stddev = log_stddev

    @classmethod
    def from_tensors(cls, parameters, deterministic):
        self = cls(shape=None, min_value=None, max_value=None)
        self.distribution = (self.mean, self.log_stddev) = parameters
        self.deterministic = deterministic
        return self

    def create_tf_operations(self, x, deterministic):
        flat_size = util.prod(self.shape)
        if isinstance(self.mean, float):
            bias = [self.mean for _ in range(flat_size)]
        else:
            bias = self.mean
        self.mean = layers['linear'](x=x, size=flat_size, bias=bias)
        self.mean = tf.reshape(tensor=self.mean, shape=((-1,) + self.shape))
        # self.mean = tf.squeeze(input=self.mean, axis=1)
        if isinstance(self.log_stddev, float):
            bias = [self.log_stddev for _ in range(flat_size)]
        else:
            bias = self.log_stddev
        self.log_stddev = layers['linear'](x=x, size=flat_size, bias=bias)
        self.log_stddev = tf.reshape(tensor=self.log_stddev, shape=((-1,) + self.shape))
        # self.log_stddev = tf.squeeze(input=self.log_stddev, axis=1)
        self.log_stddev = tf.minimum(x=self.log_stddev, y=10.0)  # prevent infinity when exp
        self.distribution = (self.mean, self.log_stddev)
        self.deterministic = deterministic

    def sample(self):
        deterministic = self.mean
        stddev = tf.exp(x=self.log_stddev)
        sampled = self.mean + stddev * tf.random_normal(shape=tf.shape(self.mean))
        return tf.where(condition=self.deterministic, x=deterministic, y=sampled)

    def log_probability(self, action):
        l2_dist = tf.square(x=(action - self.mean))
        sqr_stddev = tf.square(x=tf.exp(x=self.log_stddev))
        return -l2_dist / tf.maximum(2 * sqr_stddev, util.epsilon) - 0.5 * tf.log(x=(2 * np.pi)) - self.log_stddev

    def entropy(self):
        return self.log_stddev + 0.5 * tf.log(x=(2 * np.pi * np.e))

    def kl_divergence(self, other):
        assert isinstance(other, Gaussian)
        l2_dist = tf.square(self.mean - other.mean)
        stddev1 = tf.exp(x=self.log_stddev)
        sqr_stddev2 = tf.square(x=tf.exp(x=other.log_stddev))
        return self.log_stddev - other.log_stddev + (stddev1 + l2_dist) / tf.maximum(2 * sqr_stddev2, util.epsilon) - 0.5
