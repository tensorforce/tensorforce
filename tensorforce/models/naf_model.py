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
        self.next_state = tf.placeholder(tf.float32, (None, None) + config.state_shape, name='next_state')
        self.terminal = tf.placeholder(tf.float32, (None, None), name='terminal')
        self.reward = tf.placeholder(tf.float32, (None, None), name='reward')
        self.q_target = tf.placeholder(tf.float32, (None, None), name='q_target')
        self.episode = 0

        # Get hidden layers from network generator, then add NAF outputs, same for target network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(self.network, inputs=[self.state], episode_length=self.episode_length)
            self.training_internal_states = self.training_network.internal_inits
            self.training_v, self.mu, self.advantage, self.q, self.training_output_vars = self.create_outputs(
            self.training_network.output, 'outputs_training', config)

        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(self.network, inputs=[self.next_state], episode_length=self.episode_length)
            self.target_internal_states = self.target_network.internal_inits
            self.target_v, _, _, _, self.target_output_vars = self.create_outputs(self.target_network.output, 'outputs_target', config)

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
        """
        Creates NAF specific outputs.

        :param last_hidden_layer: Points to last hidden layer
        :param scope: TF name scope

        :return Output variables and all TF variables created in this scope
        """

        with tf.name_scope(scope):
            # State-value function
            v = layers['linear'](x=last_hidden_layer, size=1)
                                          # 'weights_regularizer_args': [config.weights_regularizer_args]},
            v = tf.reshape(v, [-1, 1])

            # Action outputs
            mu = layers['linear'](x=last_hidden_layer, size=self.num_actions)
                         # 'weights_regularizer_args': [config.weights_regularizer_args]})
            mu = tf.reshape(mu, [-1, config.actions])

            # Advantage computation
            # Network outputs entries of lower triangular matrix L
            lower_triangular_size = int(config.actions * (config.actions + 1) / 2)

            l_entries = linear(last_hidden_layer, {'num_outputs': lower_triangular_size,
                                                   'weights_regularizer': config.weights_regularizer})
                                                   # 'weights_regularizer_args': [config.weights_regularizer_args]})

            # Reshape from (?, ?, lower_triangular_size)
            l_entries = tf.reshape(l_entries, [-1, lower_triangular_size])


            # Iteratively construct matrix. Extra verbose comment here
            l_rows = []
            offset = 0

            for i in xrange(config.actions):
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

    def get_action(self, state, episode=1):
        """
        Returns naf action(s) as given by the mean output of the network.

        :param state: Current state
        :param episode: Current episode
        :return: action
        """
        fetches = [self.mu]
        fetches.extend(self.training_internal_states)
        fetches.extend(self.target_internal_states)

        feed_dict = {self.episode_length: [1], self.state: [(state, )]}

        feed_dict.update({training_internal_state: self.training_network.internal_state_inits[n] for n, training_internal_state in
                          enumerate(self.training_network.internal_state_inputs)})

        feed_dict.update({target_internal_state: self.target_network.internal_state_inits[n] for n, target_internal_state in
                          enumerate(self.target_network.internal_state_inputs)})

        print('feed dict)')
        for e in feed_dict.items():
            print(e)

        print('fetches list)')
        for e in fetches:
            print(e)

        fetched = self.session.run(fetches, feed_dict)

        action = fetched[0][0] + self.exploration(episode, self.total_states)

        # Update optional internal states, e.g. LSTM cells)
        self.training_internal_states = fetched[1:len(self.training_internal_states)]
        self.target_internal_states = fetched[1 + len(self.training_internal_states):]

        self.total_states += 1

        return action

    def update(self, batch):
        """
        Executes a NAF update on a training batch.

        :param batch:=
        :return:
        """
        float_terminals = batch['terminals'].astype(float)

        q_targets = batch['rewards'] + (1. - float_terminals) * self.discount * \
                                       self.get_target_value_estimate(batch['next_states'])

        feed_dict = {
            self.episode_length: [len(batch['rewards'])],
            self.q_targets: q_targets,
            self.actions: [batch['actions']],
            self.state: [batch['states']]}

        fetches = [self.optimize_op, self.loss, self.training_v, self.advantage, self.q]
        fetches.extend(self.training_network.internal_outputs)
        fetches.extend(self.target_network.internal_outputs)

        for n, internal_state in enumerate(self.training_network.internal_inputs):
            feed_dict[internal_state] = self.training_internal_states[n]

        for n, internal_state in enumerate(self.target_network.internal_inputs):
            feed_dict[internal_state] = self.target_internal_states[n]

        fetched = self.session.run(fetches, feed_dict)

        self.training_internal_states = fetched[5:5 + len(self.training_internal_states)]
        self.target_internal_states = fetched[5 + len(self.training_internal_states):]

    def get_target_value_estimate(self, next_states):
        """
        Estimate of next state V value through target network.

        :param next_states:
        :return:
        """

        return self.session.run(self.target_v, {self.next_states: [next_states]})

    def update_target_network(self):
        """
        Updates target network.

        :return:
        """
        self.session.run(self.target_network_update)
