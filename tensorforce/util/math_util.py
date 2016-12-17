# Copyright 2016 reinforce.io. All Rights Reserved.
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
Contains various mathematical utility functions used for policy gradient methods.
"""
import numpy as np
import tensorflow as tf


def get_log_prob_gaussian(action_dist_mean, log_std, actions):
    """

    :param action_dist_mean: Mean of action distribution
    :param log_std: Log standard deviations
    :param actions: Actions for which to compute log probabilities.
    :return:
    """
    probability = -tf.square(actions - action_dist_mean) / (2 * tf.exp(2 * log_std)) \
                  - 0.5 * tf.log(tf.constant(2 * np.pi)) - log_std

    # Sum logs
    return tf.reduce_sum(probability, 1)


def get_kl_divergence_gaussian(mean_a, log_std_a, mean_b, log_std_b):
    """
    Kullback-Leibler divergence between Gaussians a and b.

    :param mean_a: Mean of first Gaussian
    :param log_std_a: Log std of first Gaussian
    :param mean_b: Mean of second Gaussian
    :param log_std_b: Log std of second Gaussian
    :return:
    """
    exp_std_a = tf.exp(2 * log_std_a)
    exp_std_b = tf.exp(2 * log_std_b)

    return tf.reduce_sum(log_std_b - log_std_a
                         + (exp_std_a + tf.square(mean_a - mean_b)) / (2 * exp_std_b) - 0.5)


def get_entropy_gaussian(log_std):
    """
    Gaussian entropy, does not depend on mean.

    :param log_std:
    :return:
    """
    return tf.reduce_sum(log_std + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))
