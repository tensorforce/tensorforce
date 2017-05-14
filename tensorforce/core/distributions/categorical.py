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
Categorial one hot policy, used for discrete policy gradients.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):

    @classmethod
    def kl_divergence(cls, distribution1, distribution2):
        # Why do we reshape here? To go from per episode separate KL divergence to per batch KL divergence
        prob_a = tf.reshape(distribution1['policy_output'], [-1])
        prob_b = tf.reshape(distribution2['policy_output'], [-1])

        # Need to ensure numerical stability
        return tf.reduce_sum(prob_a * tf.log((prob_a + cls.epsilon) / (prob_b + cls.epsilon)), axis=[0])

    @classmethod
    def entropy(cls, distribution):
        prob = tf.reshape(distribution['policy_output'], [-1])
        return -tf.reduce_sum(prob * tf.log(prob + cls.epsilon), axis=[0])

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def create_tf_operations(self, x, sample=True):
        logits = layers['linear'](x=x, size=self.num_actions)
        self.distribution = tf.nn.softmax(logits=logits)
        if sample:
            self.value = tf.squeeze(input=tf.multinomial(logits=logits, num_samples=1), axis=1)
        else:
            self.value = tf.argmax(input=self.distribution, axis=1)

    def log_probability(self, action):
        action = tf.one_hot(indices=action, depth=self.distribution.get_shape()[1].value)
        prob = tf.reduce_sum(input_tensor=tf.multiply(x=self.distribution, y=action), axis=1)
        log_prob = tf.log(x=(prob + self.__class__.epsilon))
        return log_prob
