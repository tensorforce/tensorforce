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
            self.mean, self.log_std_dev = self.distribution

    def create_tf_operations(self, x, deterministic, min_value=None, max_value=None, **kwargs):
        assert (min_value is None) == (max_value is None)
        self.mean = tf.squeeze(input=layers['linear'](x=x, size=1), axis=1)
        if min_value is not None:  # TODO: min_value, max_value
            self.mean = min_value + tf.sigmoid(x=self.mean) * (max_value - min_value)
        self.log_std_dev = tf.squeeze(input=layers['linear'](x=x, size=1), axis=1)
        self.log_std_dev = tf.minimum(x=self.log_std_dev, y=10.0)
        self.distribution = (self.mean, self.log_std_dev)

        self.deterministic_value = self.mean
        self.sampled_value = self.mean + tf.exp(x=self.log_std_dev) * tf.random_normal(shape=tf.shape(self.mean))
        # TODO: clipping?
        # if min_value is not None:
        #     self.sampled_value = tf.clip_by_value(
        #         t=self.sampled_value,
        #         clip_value_min=min_value,
        #         clip_value_max=max_value
        #     )
        self.value = tf.where(
            condition=deterministic,
            x=self.deterministic_value,
            y=self.sampled_value
        )

    def log_probability(self, action):
        l2_dist = tf.square(action - self.mean)
        sqr_std_dev = tf.square(x=tf.exp(x=self.log_std_dev))
        log_prob = -l2_dist / (2 * sqr_std_dev + util.epsilon) - 0.5 * tf.log(tf.constant(2 * np.pi)) - self.log_std_dev
        return log_prob

    def entropy(self):
        entropy = tf.reduce_mean(self.log_std_dev + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32), axis=0)
        return entropy

    def kl_divergence(self, other):
        assert isinstance(other, Gaussian)
        l2_dist = tf.square(self.mean - other.mean)
        std_dev1 = tf.exp(x=self.log_std_dev)
        sqr_std_dev2 = tf.square(x=tf.exp(x=other.log_std_dev))
        kl_div = tf.reduce_mean(self.log_std_dev - other.log_std_dev + (std_dev1 + l2_dist) / (2 * sqr_std_dev2 + util.epsilon) - 0.5, axis=0)
        return kl_div
