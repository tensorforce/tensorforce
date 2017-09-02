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
CNN baseline value function.
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


class CNNBaseline(Baseline):

    def __init__(self, sizes, epochs=1, update_batch_size=64, learning_rate=0.001):
        """CNN baseline value function.

        Args:
            sizes: Number of neurons per hidden layer
            repeat_update: Epochs over the training data to fit the baseline
        """

        self.sizes = sizes
        self.epochs = epochs
        self.update_batch_size = update_batch_size
        self.learning_rate = learning_rate
        self.session = None

    def create_tf_operations(self, state, batch_size, scope='cnn_baseline'):

        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=(None, util.prod(state.shape)))
            self.returns = tf.placeholder(dtype=tf.float32, shape=(None,))
            self.updates = int(batch_size / self.update_batch_size) * self.epochs
            self.batch_size = batch_size

            layers = []
            for size in self.sizes:
                layers.append({'type': 'conv2d', 'size': size, 'stride': 1, 'window': 3})

            # First layer has larger window
            layers[0]['window'] = 5

            # TODO append maxpooling
            layers.append({'type': 'linear', 'size': 1})

            network = NeuralNetwork(network_builder=layered_network_builder(layers),
                                    inputs=dict(state=self.state))

            self.prediction = network.output
            loss = tf.nn.l2_loss(self.prediction - self.returns)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.optimize = optimizer.minimize(loss)

    def predict(self, states):
        return self.session.run(self.prediction, {self.state: states})[0]

    def update(self, states, returns):
        returns = np.asarray(returns)

        for _ in xrange(self.updates):
            indices = np.random.randint(low =0, high=self.batch_size, size=self.update_batch_size)
            batch_returns = returns.take(indices)

            self.session.run(self.optimize, {self.state: states, self.returns: batch_returns})



