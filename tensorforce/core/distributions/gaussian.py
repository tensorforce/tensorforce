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

    def __init__(self, distribution=None):
        assert distribution is None or len(distribution) == 2
        super(Gaussian, self).__init__(distribution)
        if self.distribution is not None:
            self.mean, self.std_dev = self.distribution

    def create_tf_operations(self, x, sample=True):
        self.mean = tf.squeeze(input=layers['linear'](x=x, size=1), axis=1)
        # self.std_dev = tf.exp(tf.squeeze(input=layers['linear'](x=x, size=1), axis=1))
        # self.std_dev = tf.ones_like(self.mean) * tf.exp(tf.Variable(initial_value=tf.random_normal(shape=(), stddev=0.01)))
        self.std_dev = tf.ones_like(self.mean)
        self.distribution = (self.mean, self.std_dev)
        if sample:
            self.value = self.mean + tf.multiply(x=self.std_dev, y=tf.random_normal(shape=tf.shape(self.mean)))
        else:
            self.value = self.mean

    def log_probability(self, action):
        l2_dist = tf.square(action - self.mean)
        sqr_std_dev = tf.square(self.std_dev)
        log_std_dev = tf.log(self.std_dev + util.epsilon)
        log_prob = -l2_dist / (2 * sqr_std_dev + util.epsilon) - 0.5 * tf.log(tf.constant(2 * np.pi)) - log_std_dev
        return log_prob

    def entropy(self):
        log_std_dev = tf.log(self.std_dev + util.epsilon)
        entropy = tf.reduce_mean(log_std_dev + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32), axis=0)

        return entropy

    def kl_divergence(self, other):
        assert isinstance(other, Gaussian)
        log_std_dev1 = tf.log(self.std_dev + util.epsilon)
        log_std_dev2 = tf.log(other.std_dev + util.epsilon)
        l2_dist = tf.square(self.mean - other.mean)
        sqr_std_dev = tf.square(other.std_dev)
        kl_div = tf.reduce_mean(log_std_dev1 - log_std_dev2 + (self.std_dev + l2_dist) / (2 * sqr_std_dev + util.epsilon) - 0.5, axis=0)
        return kl_div
