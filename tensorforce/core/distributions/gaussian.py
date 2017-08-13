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

from math import log
import numpy as np
import tensorflow as tf

from tensorforce import util
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Gaussian(Distribution):

    def __init__(self, shape, mean=0.0, stddev=1.0, logger=None):
        # if min_value is not None or max_value is not None:
        #     raise TensorForceError('Min/max value not allowed for Gaussian.')
        self.shape = shape
        self.mean = mean
        self.log_stddev = log(stddev)

    @classmethod
    def from_tensors(cls, tensors, deterministic):
        self = cls(shape=None)
        self.distribution = (self.mean, self.log_stddev) = tensors
        self.stddev = tf.exp(x=self.log_stddev)
        self.deterministic = deterministic
        return self

    def get_tensors(self):
        return (self.mean, self.log_stddev)

    def create_tf_operations(self, x, deterministic):
        self.deterministic = deterministic

        # Flat mean and log standard deviation
        flat_size = util.prod(self.shape)
        self.mean = layers['linear'](x=x, size=flat_size, bias=self.mean)
        self.log_stddev = layers['linear'](x=x, size=flat_size, bias=self.log_stddev)

        # Reshape mean and log stddev to action shape
        shape = (-1,) + self.shape
        self.mean = tf.reshape(tensor=self.mean, shape=shape)
        self.log_stddev = tf.reshape(tensor=self.log_stddev, shape=shape)

        # Clip log stddev for numerical stability
        log_eps = log(util.epsilon)
        self.log_stddev = tf.clip_by_value(t=self.log_stddev, clip_value_min=log_eps, clip_value_max=-log_eps)

        # Standard deviation
        self.stddev = tf.exp(x=self.log_stddev)

    def sample(self):
        # Deterministic: mean as action
        deterministic = self.mean
        # Non-deterministic: sample action using default normal distribution
        normal = tf.random_normal(shape=tf.shape(input=self.mean))
        sampled = self.mean + self.stddev * normal
        return tf.where(condition=self.deterministic, x=deterministic, y=sampled)

    def log_probability(self, action):
        sq_mean_distance = tf.square(x=(action - self.mean))
        sq_stddev = tf.square(x=self.stddev)

        return -0.5 * tf.log(x=(2.0 * np.pi)) - self.log_stddev - 0.5 * sq_mean_distance / sq_stddev

    def entropy(self):
        return self.log_stddev + 0.5 * tf.log(x=(2.0 * np.pi * np.e))

    def kl_divergence(self, other):
        assert isinstance(other, Gaussian)
        log_stddev_ratio = other.log_stddev - self.log_stddev
        sq_mean_distance = tf.square(x=(self.mean - other.mean))
        sq_stddev1 = tf.square(x=self.stddev)
        sq_stddev2 = tf.square(x=other.stddev)

        return log_stddev_ratio + 0.5 * (sq_stddev1 + sq_mean_distance) / sq_stddev2 - 0.5
