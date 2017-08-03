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

from tensorforce import util
from tensorforce.core.networks import layers
from tensorforce.core.distributions import Distribution


class Categorical(Distribution):

    def __init__(self, shape, num_actions, probabilities=None):
        self.shape = shape
        self.num_actions = num_actions
        if probabilities is None:
            self.logits = 0.0
        else:
            self.logits = [log(prob) for _ in range(util.prod(self.shape)) for prob in probabilities]

    @classmethod
    def from_tensors(cls, parameters, deterministic):
        self = cls(shape=None, num_actions=None)
        self.distribution = (self.logits,) = parameters
        self.probabilities = tf.exp(x=self.logits)  # assuming normalized logits
        self.num_actions = parameters[0].shape[1].value
        self.deterministic = deterministic
        return self

    def create_tf_operations(self, x, deterministic):
        # flat logits
        flat_size = util.prod(self.shape) * self.num_actions
        self.logits = layers['linear'](x=x, size=flat_size, bias=self.logits)

        # reshape logits to action shape
        shape = (-1,) + self.shape + (self.num_actions,)
        self.logits = tf.reshape(tensor=self.logits, shape=shape)

        # linearly shift logits for numerical stability
        self.logits -= tf.reduce_max(input_tensor=self.logits, axis=-1, keep_dims=True)

        # softmax for corresponding probabilities
        self.probabilities = tf.nn.softmax(logits=self.logits, dim=-1)

        # "normalized" logits
        self.logits = tf.log(x=self.probabilities)

        # general distribution values
        self.distribution = (self.logits,)
        self.deterministic = deterministic

    def sample(self):
        # deterministic: maximum likelihood action
        deterministic = tf.argmax(input=self.logits, axis=-1)
        # non-deterministic: sample action using Gumbel distribution
        uniform = tf.random_uniform(shape=tf.shape(input=self.logits), minval=util.epsilon, maxval=(1.0 - util.epsilon))
        gumbel_distribution = -tf.log(x=-tf.log(x=uniform))
        sampled = tf.argmax(input=(self.logits + gumbel_distribution), axis=-1)
        return tf.where(condition=self.deterministic, x=deterministic, y=sampled)

    def log_probability(self, action):
        one_hot = tf.one_hot(indices=action, depth=self.num_actions)
        return tf.reduce_sum(input_tensor=(self.logits * one_hot), axis=-1)

    def entropy(self):
        return -tf.reduce_sum(input_tensor=(self.probabilities * self.logits), axis=-1)

    def kl_divergence(self, other):
        assert isinstance(other, Categorical)
        log_prob_ratio = self.logits - other.logits
        return tf.reduce_sum(input_tensor=(self.probabilities * log_prob_ratio), axis=-1)
