# Copyright 2016 reinforce.io. All Rights Reserved.
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

for the update logic.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import get_variables

from tensorforce.config import create_config
from tensorforce.neural_networks.layers import dense
from tensorforce.neural_networks import NeuralNetwork
from tensorforce.util.experiment_util import global_seed
from tensorforce.util.exploration_util import exploration_mode
from tensorforce.updater import Model


class NormalizedAdvantageFunctions(Model):
    default_config = {
        'tau': 0.9,
        'epsilon': 0.1,
        'gamma': 0.95,
        'alpha': 0.005,
        'clip_gradients': False
    }

    def __init__(self, config):
        """
        Training logic for NAFs.

        :param config: Configuration parameters
        """
        super(NormalizedAdvantageFunctions, self).__init__(config)
        self.config = create_config(config, default=self.default_config)
        self.action_count = self.config.actions
        self.tau = self.config.tau
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.alpha = self.config.alpha
        self.batch_size = self.config.batch_size

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.exploration = exploration_mode[self.config.exploration_mode]

        self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
        self.next_states = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape),
                                          name="next_states")
        
        self.actions = tf.placeholder(tf.float32, [None, self.action_count], name='actions')
        self.terminals = tf.placeholder(tf.float32, [None], name='terminals')
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
        self.target_network_update = []
        self.episode = 0

        # Get hidden layers from network generator, then add NAF outputs, same for target network
        scope = '' if self.config.tf_scope is None else self.config.tf_scope + '-'
        self.training_model = NeuralNetwork(self.config.network_layers, self.state, scope=scope + 'training')
        self.target_model = NeuralNetwork(self.config.network_layers, self.next_states, scope=scope + 'target')
        self.optimizer = tf.train.AdamOptimizer(self.alpha)

        # Create output fields
        self.training_v, self.mu, self.advantage, self.q, self.training_output_vars = self.create_outputs(
            self.training_model.get_output(), 'outputs_training')
        self.target_v, _, _, _, self.target_output_vars = self.create_outputs(self.target_model.get_output(),
                                                                              'outputs_target')
        self.create_training_operations()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

    def get_action(self, state, episode=1, total_states=0):
        """
        Returns naf action(s) as given by the mean output of the network.

        :param state: Current state
        :param episode: Current episode
        :param total_states: Total states processed
        :return:
        """
        action = self.session.run(self.mu, {self.state: [state]})[0]

        return action + self.exploration(self.random, self.episode)

    def update(self, batch):
        """
        Executes a NAF update on a training batch.

        :param batch:=
        :return:
        """
        float_terminals = batch['terminals'].astype(float)

        q_targets = batch['rewards'] + (1. - float_terminals) * self.gamma * np.squeeze(
            self.get_target_value_estimate(batch['next_states']))

        self.session.run([self.optimize_op, self.loss, self.training_v, self.advantage, self.q], {
            self.q_targets: q_targets,
            self.actions: batch['actions'],
            self.state: batch['states']})

    def create_outputs(self, last_hidden_layer, scope):
        """
        Creates NAF specific outputs.

        :param last_hidden_layer: Points to last hidden layer
        :param scope: TF name scope

        :return Output variables and all TF variables created in this scope
        """

        with tf.name_scope(scope):
            # State-value function
            v = dense(last_hidden_layer, {'neurons': 1, 'regularization': self.config.regularizer,
                                          'regularization_param': self.config.regularization_param}, scope + 'v')

            # Action outputs
            mu = dense(last_hidden_layer, {'neurons': self.action_count, 'regularization': self.config.regularizer,
                                           'regularization_param': self.config.regularization_param}, scope + 'mu')

            # Advantage computation
            # Network outputs entries of lower triangular matrix L
            lower_triangular_size = int(self.action_count * (self.action_count + 1) / 2)
            l_entries = dense(last_hidden_layer, {'neurons': lower_triangular_size,
                                                  'regularization': self.config.regularizer,
                                                  'regularization_param': self.config.regularization_param},
                              scope + 'l')

            # Iteratively construct matrix. Extra verbose comment here
            l_rows = []
            offset = 0

            for i in xrange(self.action_count):
                # Diagonal elements are exponentiated, otherwise gradient often 0
                # Slice out lower triangular entries from flat representation through moving offset

                diagonal = tf.exp(tf.slice(l_entries, (0, offset), (-1, 1)))

                n = self.action_count - i - 1
                # Slice out non-zero non-diagonal entries, - 1 because we already took the diagonal
                non_diagonal = tf.slice(l_entries, (0, offset + 1), (-1, n))

                # Fill up row with zeros
                row = tf.pad(tf.concat(1, (diagonal, non_diagonal)), ((0, 0), (i, 0)))
                offset += (self.action_count - i)
                l_rows.append(row)

            # Stack rows to matrix
            l_matrix = tf.transpose(tf.pack(l_rows, axis=1), (0, 2, 1))

            # P = LL^T
            p_matrix = tf.batch_matmul(l_matrix, tf.transpose(l_matrix, (0, 2, 1)))

            # Need to adjust dimensions to multiply with P.
            action_diff = tf.expand_dims(self.actions - mu, -1)

            # A = -0.5 (a - mu)P(a - mu)
            advantage = -0.5 * -tf.batch_matmul(tf.transpose(action_diff, [0, 2, 1]),
                                                tf.batch_matmul(p_matrix, action_diff))

            with tf.name_scope('q_values'):
                # Q = A + V
                q_value = v + advantage

        # Get all variables under this scope for target network update
        return v, mu, advantage, q_value, get_variables(scope)

    def create_training_operations(self):
        """
        NAF update logic.
        """

        with tf.name_scope("update"):
            self.q_targets = tf.placeholder(tf.float32, [None], name='q_targets')

            # MSE
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_targets, tf.squeeze(self.q)),
                                       name='compute_surrogate_loss')
            self.optimize_op = self.optimizer.minimize(self.loss)

        with tf.name_scope("update_target"):
            # Combine hidden layer variables and output layer variables
            self.training_vars = self.training_model.get_variables() + self.training_output_vars
            self.target_vars = self.target_model.get_variables() + self.target_output_vars

            for v_source, v_target in zip(self.training_vars, self.target_vars):
                update = v_target.assign_sub(self.tau * (v_target - v_source))

                self.target_network_update.append(update)

    def get_target_value_estimate(self, next_states):
        """
        Estimate of next state V value through target network.

        :param next_states:
        :return:
        """

        return self.session.run(self.target_v, {self.next_states: next_states})


    def update_target_network(self):
        """
        Updates target network.

        :return:
        """
        self.session.run(self.target_network_update)
