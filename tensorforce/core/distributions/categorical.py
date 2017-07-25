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

from math import log
import tensorflow as tf

from tensorforce import util, TensorForceError
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):

    def __init__(self, shape, num_actions, probabilities=None):
        self.shape = shape
        self.num_actions = num_actions
        if num_actions is None:
            return
        elif probabilities is None:
            self.probabilities = [0.0 for _ in range(num_actions)]
        elif len(probabilities) == num_actions:
            self.probabilities = [log(prob) for prob in probabilities]
        else:
            raise TensorForceError('Invalid length for probabilities.')

    @classmethod
    def from_tensors(cls, parameters, deterministic):
        self = cls(shape=None, num_actions=None)
        self.distribution = (self.probabilities,) = parameters
        self.num_actions = parameters[0].shape[1].value
        self.deterministic = deterministic
        return self

    def create_tf_operations(self, x, deterministic):
        flat_size = util.prod(self.shape) * self.num_actions
        if len(self.probabilities) < flat_size:
            bias = [prob for _ in range(util.prod(self.shape)) for prob in self.probabilities]
        else:
            bias = self.probabilities
        logits = layers['linear'](x=x, size=flat_size, bias=bias)
        logits = tf.reshape(tensor=logits, shape=((-1,) + self.shape + (self.num_actions,)))
        self.probabilities = tf.nn.softmax(logits=logits)
        self.distribution = (self.probabilities,)
        self.deterministic = deterministic

    def sample(self):
        deterministic = tf.argmax(input=self.probabilities, axis=-1)
        logits = tf.log(x=self.probabilities)
        logits = tf.reshape(tensor=logits, shape=(-1, self.num_actions))
        sampled = tf.squeeze(input=tf.multinomial(logits=logits, num_samples=1), axis=1)
        sampled = tf.reshape(tensor=sampled, shape=(-1,) + self.shape)
        return tf.where(condition=self.deterministic, x=deterministic, y=sampled)

    def log_probability(self, action):
        action = tf.one_hot(indices=action, depth=self.num_actions)
        prob = tf.reduce_sum(input_tensor=tf.multiply(x=self.probabilities, y=action), axis=-1)
        log_prob = tf.log(x=tf.maximum(prob, util.epsilon))
        return log_prob

    def entropy(self):
        return -tf.reduce_sum(self.probabilities * tf.log(x=tf.maximum(self.probabilities, util.epsilon)), axis=-1)

    def kl_divergence(self, other):
        assert isinstance(other, Categorical)
        # Need to ensure numerical stability
        return tf.reduce_sum(self.probabilities * tf.log(self.probabilities / tf.maximum(other.probabilities, util.epsilon)), axis=-1)
