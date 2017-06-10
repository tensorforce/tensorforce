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

from tensorforce import util
from tensorforce.core.networks import NeuralNetwork, layered_network_builder
from tensorforce.core.value_functions import ValueFunction


class MLPValueFunction(ValueFunction):

    def __init__(self, session, num_hidden, repeat_update=100):
        super(MLPValueFunction, self).__init__(session)
        self.num_hidden = num_hidden
        self.repeat_update = repeat_update

    def create_tf_operations(self, config):
        if len(config.states) > 1:
            raise Exception()

        with tf.variable_scope('mlp_value_function'):
            self.state = tf.placeholder(dtype=tf.float32, shape=(None, util.prod(config.states[0]['shape'])))
            self.returns = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            network_builder = layered_network_builder((
                {'type': 'dense', 'size': self.num_hidden},
                {'type': 'dense', 'size': 1}))

            network = NeuralNetwork(network_builder=network_builder, inputs=[self.state])

            self.prediction = network.output
            loss = tf.nn.l2_loss(self.prediction - self.returns)

            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            self.optimize = optimizer.minimize(loss)

    def predict(self, states):
        """Predicts return of state(s), V(s)
        
        Args:
            states: 

        Returns: 

        """

        states = next(iter(states.values()))
        return self.session.run(self.prediction, {self.state: states})[0]

    def update(self, states, returns):
        states = next(iter(states.values()))

        for _ in range(self.repeat_update):
            self.session.run(self.optimize, {self.state: states, self.returns: returns})
