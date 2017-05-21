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
Implements normalized advantage functions, largely following

https://github.com/carpedm20/NAF-tensorflow/blob/master/src/network.py

for the update logic with different modularisation.

The core training update code is under MIT license, for more information see LICENSE-EXT.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import xrange
from tensorflow.contrib.framework import get_variables

from tensorforce.core import Model
from tensorforce.core.networks import NeuralNetwork, layers
from tensorforce.core.networks.layers import linear_layer


class NAFModel(Model):

    default_config = {
        'tau': 0.001
    }

    def __init__(self, config, network_builder):
        """
        Training logic for NAFs.

        :param config: Configuration parameters
        """
        config.default(NAFModel.default_config)
        self.network = network_builder
        super(NAFModel, self).__init__(config)


    def create_tf_operations(self, config):
        super(NAFModel, self).create_tf_operations(config)

        # placeholders
        self.terminal = tf.placeholder(tf.float32, (None, None), name='terminal')
        self.reward = tf.placeholder(tf.float32, (None, None), name='reward')
        self.q_target = tf.placeholder(tf.float32, (None, None), name='q_target')
        self.episode = 0

        # Get hidden layers from network generator, then add NAF outputs, same for target network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(self.network,
                                                  inputs={name: state for name, state in self.state.items()})

            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)
            training_output = dict()

            for action in self.action:
                self.training_v, self.mu, self.advantage, self.q, self.training_output_vars = self.create_outputs(
                    self.training_network.output, 'outputs_training', config)
                self.action_taken[action] = self.mu

        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(self.network, inputs={name: state for name, state in self.state.items()})
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)
            target_value = dict()


            for action in self.action:
                self.target_v, target_mu, _, _, self.target_output_vars = self.create_outputs(self.target_network.output, 'outputs_target', config)
                target_value[action] = target_mu

        # NAF update logic
        with tf.name_scope("update"):
            # MSE
            loss = tf.reduce_mean(tf.squared_difference(self.q_target, tf.squeeze(self.q)), name='loss')
            tf.losses.add_loss(loss)

        with tf.name_scope("update_target"):
            # Combine hidden layer variables and output layer variables
            self.training_vars = self.training_network.variables + self.training_output_vars
            self.target_vars = self.target_network.variables + self.target_output_vars

            self.target_network_update = []
            for v_source, v_target in zip(self.training_vars, self.target_vars):
                update = v_target.assign_sub(config.tau * (v_target - v_source))
                self.target_network_update.append(update)

    def create_outputs(self, last_hidden_layer, scope, config):
        """Creates NAF specific outputs.
        
        Args:
            last_hidden_layer: 
            scope: 
            config: 

        Returns:

        """

        with tf.name_scope(scope):
            # State-value function
            v = layers['linear'](x=last_hidden_layer, size=1)

            # Action outputs
            mu = layers['linear'](x=last_hidden_layer, size=config.num_actions)

            # Advantage computation
            # Network outputs entries of lower triangular matrix L
            lower_triangular_size = int(config.actions * (config.actions + 1) / 2)

            l_entries = linear_layer(last_hidden_layer, {'num_outputs': lower_triangular_size,
                                                   'weights_regularizer': config.weights_regularizer})

            # Iteratively construct matrix. Extra verbose comment here
            l_rows = []
            offset = 0

            for i in xrange(config.num_actions):
                # Diagonal elements are exponentiated, otherwise gradient often 0
                # Slice out lower triangular entries from flat representation through moving offset

                diagonal = tf.exp(tf.slice(l_entries, (0, offset), (-1, 1)))

                n = config.actions - i - 1
                # Slice out non-zero non-diagonal entries, - 1 because we already took the diagonal
                non_diagonal = tf.slice(l_entries, (0, offset + 1), (-1, n))

                # Fill up row with zeros
                row = tf.pad(tf.concat(axis=1, values=(diagonal, non_diagonal)), ((0, 0), (i, 0)))
                offset += (config.actions - i)
                l_rows.append(row)

            # Stack rows to matrix
            l_matrix = tf.transpose(tf.stack(l_rows, axis=1), (0, 2, 1))

            # P = LL^T
            p_matrix = tf.matmul(l_matrix, tf.transpose(l_matrix, (0, 2, 1)))

            # Need to adjust dimensions to multiply with P.
            # TODO see if this can be done simpler
            actions = tf.reshape(self.action, [-1, config.actions])
            action_diff = tf.expand_dims(actions - mu, -1)

            # A = -0.5 (a - mu)P(a - mu)
            advantage = -0.5 * tf.matmul(tf.transpose(action_diff, [0, 2, 1]),
                                         tf.matmul(p_matrix, action_diff))
            advantage = tf.reshape(advantage, [-1, 1])

            with tf.name_scope('q_values'):
                # Q = A + V
                q_value = v + advantage

        # Get all variables under this scope for target network update
        return v, mu, advantage, q_value, get_variables(scope)

    def update_target_network(self):
        """
        Updates target network.

        :return:
        """
        self.session.run(self.target_network_update)
