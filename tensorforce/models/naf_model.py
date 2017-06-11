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

    default_config = dict(
        update_target_weight=1.0,
        clip_gradients=0.0
    )
    allows_discrete_actions = False
    allows_continuous_actions = True

    def __init__(self, config):
        """
        Training logic for NAFs.

        :param config: Configuration parameters
        """
        config.default(NAFModel.default_config)
        super(NAFModel, self).__init__(config)

    def create_tf_operations(self, config):
        super(NAFModel, self).create_tf_operations(config)

        # Get hidden layers from network generator, then add NAF outputs, same for target network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(config.network, inputs=self.state)
            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)

        with tf.variable_scope('training_outputs'):
            num_actions = len(self.action)
            # Action outputs
            mean = layers['linear'](x=self.training_network.output, size=num_actions)
            for n, action in enumerate(sorted(self.action)):
                # mean = tf.Print(mean,[mean])
                self.action_taken[action] = mean[n]

            # Advantage computation
            # Network outputs entries of lower triangular matrix L
            lower_triangular_size = num_actions * (num_actions + 1) // 2
            l_entries = layers['linear'](x=self.training_network.output, size=lower_triangular_size)

            l_matrix = tf.exp(tf.map_fn(tf.diag, l_entries[:, :num_actions]))

            if num_actions > 1:
                offset = num_actions
                l_columns = list()
                for zeros, size in enumerate(xrange(num_actions - 1, 0, -1), 1):
                    column = tf.pad(l_entries[:, offset: offset + size], ((0, 0), (zeros, 0)))
                    l_columns.append(column)
                    offset += size
                l_matrix += tf.stack(l_columns, 1)

            # P = LL^T
            p_matrix = tf.matmul(l_matrix, tf.transpose(l_matrix, (0, 2, 1)))
            # p_matrix = tf.Print(p_matrix, [p_matrix])

            # l_rows = []
            # offset = 0
            # for i in xrange(num_actions):
            #     # Diagonal elements are exponentiated, otherwise gradient often 0
            #     # Slice out lower triangular entries from flat representation through moving offset
            #     diagonal = tf.exp(l_entries[:, offset])  # tf.slice(l_entries, (0, offset), (-1, 1))
            #     n = config.actions - i - 1
            #     # Slice out non-zero non-diagonal entries, - 1 because we already took the diagonal
            #     non_diagonal = l_entries[:, offset + 1: offset + n + 1]  # tf.slice(l_entries, (0, offset + 1), (-1, n))
            #     # Fill up row with zeros
            #     row = tf.pad(tf.concat(axis=1, values=(diagonal, non_diagonal)), ((0, 0), (i, 0)))
            #     offset += (num_actions - i)
            #     l_rows.append(row)
            #
            # # Stack rows to matrix
            # l_matrix = tf.transpose(tf.stack(l_rows, axis=1), (0, 2, 1))

            actions = tf.stack(values=[self.action[name] for name in sorted(self.action)], axis=1)
            action_diff = actions - mean

            # A = -0.5 (a - mean)P(a - mean)
            advantage = -tf.matmul(tf.expand_dims(action_diff, 1), tf.matmul(p_matrix, tf.expand_dims(action_diff, 2))) / 2
            advantage = tf.squeeze(advantage, 2)

            # Q = A + V
            # State-value function
            value = layers['linear'](x=self.training_network.output, size=1)
            q_value = tf.squeeze(value + advantage, 1)
            training_output_vars = get_variables('training_outputs')

        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(config.network, inputs=self.state)
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)
            target_value = dict()

        with tf.variable_scope('target_outputs'):
            # State-value function
            target_value_output = layers['linear'](x=self.target_network.output, size=1)
            for action in self.action:
                # Naf directly outputs V(s)
                target_value[action] = target_value_output

            target_output_vars = get_variables('target_outputs')

        with tf.name_scope("update"):
            for action in self.action:
                q_target = self.reward[:-1] + (1.0 - tf.cast(self.terminal[:-1], tf.float32)) * config.discount\
                                              * target_value[action][1:]
                delta = q_target - q_value[:-1]

                # We observe issues with numerical stability in some tests, gradient clipping can help
                if config.clip_gradients > 0.0:
                    huber_loss = tf.where(tf.abs(delta) < config.clip_gradients, tf.multiply(tf.square(delta), 0.5),
                                          tf.abs(delta) - 0.5)
                    loss = tf.reduce_mean(huber_loss)
                else:
                    loss = tf.reduce_mean(tf.square(delta))
                # loss = tf.Print(loss, [loss])
                tf.losses.add_loss(loss)

        with tf.name_scope("update_target"):
            # Combine hidden layer variables and output layer variables
            training_vars = self.training_network.variables + training_output_vars
            target_vars = self.target_network.variables + target_output_vars

            self.target_network_update = list()
            for v_source, v_target in zip(training_vars, target_vars):
                update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                self.target_network_update.append(update)

    def update_target_network(self):
        """
        Updates target network.

        :return:
        """
        self.session.run(self.target_network_update)
