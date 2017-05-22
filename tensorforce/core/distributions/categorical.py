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

from tensorforce import util
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):

    def __init__(self, distribution=None, num_actions=None):
        assert distribution is None or len(distribution) == 1
        assert (distribution is None) != (num_actions is None)
        super(Categorical, self).__init__(distribution)
        if num_actions is None:
            num_actions = distribution[0].get_shape()[1].value
        self.num_actions = num_actions
        if self.distribution is not None:
            self.probabilities, = self.distribution

    def create_tf_operations(self, x, sample=True):
        logits = layers['linear'](x=x, size=self.num_actions)
        self.probabilities = tf.nn.softmax(logits=logits)
        self.distribution = (self.probabilities,)
        if sample:
            self.value = tf.squeeze(input=tf.multinomial(logits=logits, num_samples=1), axis=1)
        else:
            self.value = tf.argmax(input=self.distribution, axis=1)

    def log_probability(self, action):
        action = tf.one_hot(indices=action, depth=self.probabilities.get_shape()[1].value)
        prob = tf.reduce_sum(input_tensor=tf.multiply(x=self.probabilities, y=action), axis=1)
        log_prob = tf.log(x=(prob + util.epsilon))
        return log_prob

    def entropy(self):
        return -tf.reduce_sum(self.probabilities * tf.log(self.probabilities + util.epsilon), axis=[0])

    def kl_divergence(self, other):
        assert isinstance(other, Categorical)
        # Need to ensure numerical stability
        return tf.reduce_sum(self.probabilities * tf.log((self.probabilities + util.epsilon) / (other.probabilities + util.epsilon)), axis=[0])
