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
Multi-layer perceptron baseline value function
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from tensorforce.models.baselines.value_function import ValueFunction
from tensorforce.models.neural_networks import NeuralNetwork


class MLPValueFunction(ValueFunction):

    def __init__(self, session, state_size, layer_size, update_iterations=100):
        self.session = session
        self.mlp = None
        self.update_iterations = update_iterations
        self.labels = tf.placeholder(tf.float32, shape=(None, 1), name="labels")
        self.create_net(state_size=state_size, layer_size=layer_size)

    def predict(self, path):
        if self.mlp is None:
            return np.zeros(len(path["rewards"]))
        else:
            return self.session.run(self.mlp, {self.input: self.get_features(path)})

    def fit(self, paths):
        feature_matrix = np.concatenate([self.get_features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(self.update_iterations):
            self.session.run(self.update, {self.input: feature_matrix, self.labels: returns})

    def create_net(self, state_size, layer_size):
        with tf.variable_scope("mlp_value_function"):
            self.input = tf.placeholder(tf.float32, shape=[None, self.get_features_size(state_size)], name="input")

            network_builder = NeuralNetwork.layered_network((
                {'type': 'dense', 'num_outputs': layer_size},
                {'type': 'dense', 'num_outputs': 1}))
            network = NeuralNetwork(network_builder=network_builder, inputs=[self.input])

            # hidden_1 = dense(layer_input=self.input, {'num_outputs': input_shape}, scope='hidden_1')
            # hidden_2 = dense(hidden_1, {'num_outputs': self.layer_size}, scope='hidden_2')
            # out = dense(hidden_2, {'num_outputs': 1}, scope='out')
            self.mlp = tf.reshape(network.output, (-1, 1))

            l2 = tf.nn.l2_loss(self.mlp - self.labels)
            self.update = tf.train.AdamOptimizer().minimize(l2)

            self.session.run(tf.global_variables_initializer())
