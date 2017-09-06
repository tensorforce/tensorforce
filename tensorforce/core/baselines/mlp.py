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
Multi-layer perceptron baseline value function.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange
import tensorflow as tf
import numpy as np

from tensorforce import util
from tensorforce.core.networks import NeuralNetwork, layered_network_builder
from tensorforce.core.baselines import Baseline


class MLPBaseline(Baseline):

    def __init__(self, sizes, epochs=1, update_batch_size=64, learning_rate=0.001):
        """Multilayer-perceptron baseline value function.

        Args:
            sizes: Number of neurons per hidden layer
            repeat_update: Epochs over the training data to fit the baseline
        """

        self.sizes = sizes
        self.epochs = epochs
        self.update_batch_size = update_batch_size
        self.learning_rate = learning_rate
        self.session = None

    def create_tf_operations(self, state, scope='mlp_baseline'):
        with tf.variable_scope(scope) as scope:
            self.state = tf.placeholder(dtype=tf.float32, shape=(None, util.prod(state.shape)))
            self.returns = tf.placeholder(dtype=tf.float32, shape=(None,))

            layers = []
            for size in self.sizes:
                layers.append({'type': 'dense', 'size': size})

            layers.append({'type': 'linear', 'size': 1})

            network = NeuralNetwork(network_builder=layered_network_builder(layers),
                                    inputs=dict(state=self.state))

            self.prediction = tf.squeeze(input=network.output, axis=1)
            loss = tf.nn.l2_loss(self.prediction - self.returns)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            variables = tf.contrib.framework.get_variables(scope=scope)
            self.optimize = optimizer.minimize(loss, var_list=variables)

    def predict(self, states):
        return self.session.run(self.prediction, {self.state: states})

    def update(self, states, returns):
        states = np.asarray(states)
        returns = np.asarray(returns)
        batch_size = states.shape[0]
        updates = int(batch_size / self.update_batch_size) * self.epochs

        for _ in xrange(updates):
            indices = np.random.randint(low=0, high=batch_size, size=self.update_batch_size)
            batch_states = states.take(indices, axis=0)
            batch_returns = returns.take(indices, axis=0)

            self.session.run(self.optimize, {self.state: batch_states, self.returns: batch_returns})
