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


class NAFModel(Model):

    default_config = {
        'tau': 0.001
    }
    allows_discrete_actions = False
    allows_continuous_actions = True

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



        # Get hidden layers from network generator, then add NAF outputs, same for target network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(self.network,
                                                  inputs={name: state for name, state in self.state.items()})

            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)

            # Create all actions as one vector of means (mus)
            num_actions = 0
            for name, action in config.actions:
                num_actions += action.num_actions

            training_v, self.mu, advantage, self.q, training_output_vars = self.create_outputs(
                self.training_network.output, 'outputs_training', config, num_actions)

            idx = 0
            print(self.q.get_shape)
            for action in self.action:
                self.action_taken[action] = self.mu[idx]
                idx +=1

        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(self.network, inputs={name: state for name, state in self.state.items()})
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)
            target_value = dict()


            target_v, target_mu, _, _, target_output_vars = self.create_outputs(self.target_network.output, 'outputs_target', config,
                                                                                num_actions)
            for action in self.action:
                # Naf directly outputs V(s)
                target_value[action] = target_v

        with tf.name_scope("update"):
            for action in self.action:
                q_target = self.reward[:-1] + (1.0 - tf.cast(self.terminal[:-1], tf.float32)) * config.discount * target_value[action][1:]
                loss = tf.reduce_mean(tf.squared_difference(q_target, tf.squeeze(self.q)), name='loss')
                tf.losses.add_loss(loss)

        with tf.name_scope("update_target"):
            # Combine hidden layer variables and output layer variables
            self.training_vars = self.training_network.variables + training_output_vars
            self.target_vars = self.target_network.variables + target_output_vars

            self.target_network_update = []
            for v_source, v_target in zip(self.training_vars, self.target_vars):
                update = v_target.assign_sub(config.tau * (v_target - v_source))
                self.target_network_update.append(update)

    def create_outputs(self, last_hidden_layer, scope, config, num_actions):
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
            mu = layers['linear'](x=last_hidden_layer, size=num_actions)

            # Advantage computation
            # Network outputs entries of lower triangular matrix L
            lower_triangular_size = int(num_actions * (num_actions + 1) / 2)

            l_entries = layers['linear'](x=last_hidden_layer, size=lower_triangular_size)

            # Iteratively construct matrix. Extra verbose comment here
            l_rows = []
            offset = 0

            for i in xrange(num_actions):
                # Diagonal elements are exponentiated, otherwise gradient often 0
                # Slice out lower triangular entries from flat representation through moving offset

                diagonal = tf.exp(tf.slice(l_entries, (0, offset), (-1, 1)))

                n = config.actions - i - 1
                # Slice out non-zero non-diagonal entries, - 1 because we already took the diagonal
                non_diagonal = tf.slice(l_entries, (0, offset + 1), (-1, n))

                # Fill up row with zeros
                row = tf.pad(tf.concat(axis=1, values=(diagonal, non_diagonal)), ((0, 0), (i, 0)))
                offset += (num_actions - i)
                l_rows.append(row)

            # Stack rows to matrix
            l_matrix = tf.transpose(tf.stack(l_rows, axis=1), (0, 2, 1))

            # P = LL^T
            p_matrix = tf.matmul(l_matrix, tf.transpose(l_matrix, (0, 2, 1)))

            # Need to adjust dimensions to multiply with P.

            #TODO self.action?
            actions = tf.reshape(self.action, [-1, num_actions])
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
