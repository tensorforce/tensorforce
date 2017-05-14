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

from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Gaussian(Distribution):

    def kl_divergence(self, dist_a, dist_b,):
        mean_a = tf.reshape(dist_a['policy_output'], [-1])
        log_std_a = tf.reshape(dist_a['policy_log_std'], [-1])

        mean_b = tf.reshape(dist_b['policy_output'], [-1])
        log_std_b = tf.reshape(dist_b['policy_log_std'], [-1])

        exp_std_a = tf.exp(2 * log_std_a)
        exp_std_b = tf.exp(2 * log_std_b)

        return tf.reduce_sum(log_std_b - log_std_a + (exp_std_a + tf.square(mean_a - mean_b)) / (2 * exp_std_b) - 0.5, axis=[0])

    def entropy(self, dist):
        log_std = tf.reshape(dist['policy_log_std'], [-1])

        return tf.reduce_sum(log_std + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32), axis=[0])

    def create_tf_operations(self, x, sample=True):
        mean = layers['linear'](x=x, size=1)
        mean = tf.squeeze(input=mean, axis=1)
        log_std_dev = tf.Variable(initial_value=tf.random_normal(shape=(), stddev=0.01))
        self.distribution = (mean, log_std_dev)
        if sample:
            std_dev = tf.exp(x=log_std_dev)
            self.value = mean + tf.multiply(x=std_dev, y=tf.random_normal(shape=tf.shape(mean)))
        else:
            self.value = mean

    def log_probability(self, action):
        mean, log_std_dev = self.distribution
        probability = -tf.square(action - mean) / (2 * tf.exp(2 * log_std_dev)) \
                      - 0.5 * tf.log(tf.constant(2 * np.pi)) - log_std_dev
        return probability
