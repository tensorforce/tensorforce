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

from tensorforce.models.policies.distribution import Distribution
import numpy as np
import tensorflow as tf


class Categorical(Distribution):
    def __init__(self, random):
        super(Categorical, self).__init__(random)

    def kl_divergence(self, dist_a, dist_b):
        prob_a = dist_a['policy_output']
        prob_b = dist_b['policy_output']

        # Need to ensure numerical stability
        return tf.reduce_sum(prob_a * tf.log((prob_a + self.epsilon) / (prob_b + self.epsilon)))

    def entropy(self, dist):
        prob = dist['policy_output']

        return -tf.reduce_sum(prob * tf.log((prob + self.epsilon)))

    def log_prob(self, dist, actions):
        prob = dist['policy_output']

        return tf.log(tf.reduce_sum(tf.multiply(prob, actions), [1]) + self.epsilon)

    def fixed_kl(self, dist):
        """
        KL divergence with first param fixed. Used in TRPO update.

        """
        prob = dist['policy_output']

        return tf.reduce_sum(tf.stop_gradient(prob)
                             * tf.log(tf.stop_gradient(prob + self.epsilon) / (prob + self.epsilon)))

    def sample(self, dist):
        prob = dist['policy_output']

        # TODO if this repeatedly causes errors, we need to re-weigh to ensure sum to 1

        # Categorical dist is special case of multinomial
        # Renormalise for numerical stability
        prob = np.round(prob / np.sum(prob), decimals=8)
        try:
            return np.flatnonzero(self.random.multinomial(1, prob, 1))[0]
        except ValueError as e:
            print(e)
            print(np.sum(prob))
            print(prob)
