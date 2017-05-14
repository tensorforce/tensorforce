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

import numpy as np
import tensorflow as tf

from tensorforce.models.policies.distribution import Distribution


class Gaussian(Distribution):

    def __init__(self, random):
        super(Gaussian, self).__init__(random)

    def log_prob(self, dist, actions=0):
        mean = dist['policy_output']
        log_std = dist['policy_log_std']

        probability = -tf.square(actions - mean) / (2 * tf.exp(2 * log_std)) \
                      - 0.5 * tf.log(tf.constant(2 * np.pi)) - log_std

        # Sum logs
        return tf.reduce_sum(probability, axis=[2])

    def kl_divergence(self, dist_a, dist_b,):
        mean_a = tf.reshape(dist_a['policy_output'], [-1])
        log_std_a = tf.reshape(dist_a['policy_log_std'], [-1])

        mean_b = tf.reshape(dist_b['policy_output'], [-1])
        log_std_b = tf.reshape(dist_b['policy_log_std'], [-1])

        exp_std_a = tf.exp(2 * log_std_a)
        exp_std_b = tf.exp(2 * log_std_b)

        return tf.reduce_sum(log_std_b - log_std_a
                             + (exp_std_a + tf.square(mean_a - mean_b)) / (2 * exp_std_b) - 0.5, axis=[0])

    def entropy(self, dist):
        log_std = tf.reshape(dist['policy_log_std'], [-1])

        return tf.reduce_sum(log_std + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32), axis=[0])

    def fixed_kl(self, dist):
        """
        KL divergence with first param fixed.

        :param dist:
        :return:
        """
        mean = dist['policy_output']
        log_std = dist['policy_log_std']

        mean_a, log_std_a = map(tf.stop_gradient, [mean, log_std])

        dist_a = dict(policy_output=mean_a,
                      policy_log_std=log_std_a)

        return self.kl_divergence(dist_a, dist)
