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
Implements normalized advantage functions as described here:
https://arxiv.org/abs/1603.00748
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import get_variables

from tensorforce.neural_networks.layers import dense
from tensorforce.neural_networks.neural_network import get_layers, NeuralNetwork
from tensorforce.util.experiment_util import global_seed
from tensorforce.value_functions.value_function import ValueFunction


class NormalizedAdvantageFunctions(ValueFunction):
    default_config = {
        'tau': 0,
        'epsilon': 0.1,
        'gamma': 0,
        'alpha': 0.5,
        'clip_gradients': False
    }

    def __init__(self, config):
        """
        Training logic for NAFs.

        :param config: Configuration parameters
        """
        super(NormalizedAdvantageFunctions, self).__init__(config)
        self.action_count = self.config['actions']
        self.config = config
        self.tau = self.config.tau
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.alpha = self.config.alpha
        self.batch_size = self.config.batch_size

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.state = tf.placeholder(tf.float32, [None] + list(self.config.state_shape), name="state")
        self.next_states = tf.placeholder(tf.float32, [None] + list(self.config.state_shape), name="next_states")
        self.actions = tf.placeholder(tf.int64, [None], name='actions')
        self.terminals = tf.placeholder(tf.float32, [None], name='terminals')
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
        self.target_network_update = []
        self.step = 0

        # Get hidden layers from network generator, then add NAF outputs, same for target network
        self.training_model = NeuralNetwork(self.config.network_layers, self.state, 'training')
        self.target_model = NeuralNetwork(self.config.network_layers, self.state, 'model')
        self.optimizer = tf.train.AdamOptimizer(self.alpha)

        self.training_output_vars = self.create_outputs(self.training_model.get_output(), 'outputs_training')
        self.target_output_vars = self.create_outputs(self.target_model.get_output(), 'outputs_target')

    def get_noise(self, step):
        """
        Returns a noise sample from the configured exploration strategy.

        :param step:
        :return:
        """
        return self.random

    def get_action(self, state):
        """
        Returns naf actions.
        :param state:
        :return:
        """
        action = self.session.run(self.mu, {self.state: [state]})

        return action + self.get_noise(self.step)

    def update(self, batch):
        pass

    def create_outputs(self, last_hidden_layer, scope):
        """
        Creates NAF specific outputs.
        :param hidden_layers: Points to last hidden layer
        """

        with tf.name_scope(scope):
            # State-value function
            self.v = dense(last_hidden_layer, {'neurons': 1, 'regularization': self.config['regularizer'],
                                           'regularization_param': self.config['regularization_param']}, 'v')

            # Action outputs
            self.mu = dense(last_hidden_layer, {'neurons':  self.action_count, 'regularization': self.config['regularizer'],
                                            'regularization_param': self.config['regularization_param']}, 'v')

            # Advantage computation
            # Network outputs entries of lower triangular matrix L
            lower_triangular_size =  self.action_count * ( self.action_count+ 1) / 2
            self.l_entries = dense(last_hidden_layer, {'neurons': lower_triangular_size,
                                                   'regularization': self.config['regularizer'],
                                                   'regularization_param': self.config['regularization_param']}, 'v')

            # Iteratively construct matrix. Extra verbose comment here
            l_rows = []
            offset = 0

            for i in xrange(self.action_count):

                # Diagonal elements are exponentiated, otherwise gradient often 0
                # Slice out lower triangular entries from flat representation through moving offset
                diagonal = tf.exp(tf.slice(self.l_matrix, (0, offset), (-1, 1)))

                n = self.actions - i - 1
                # Slice out non-zero non-diagonal entries, - 1 because we already took the diagonal
                non_diagonal = tf.slice(self.l_matrix, (0, offset + 1), (-1, n))

                # Fill up row with n - i zeros
                row = tf.pad(tf.concat(1, (diagonal, non_diagonal)), ((0, 0), (i, 0)))
                offset += (self.actions - i)
                l_rows.append(row)

            # Stack rows to matrix
            self.l_matrix = tf.transpose(tf.pack(l_rows, axis=1), (0, 2, 1))

            # P = LL^T
            self.p_matrix = tf.batch_matmul(self.l_matrix, tf.transpose(self.l_matrix, (0, 2, 1)))

            # Need to adjust dimensions to multiply with P.
            action_diff = tf.expand_dims(self.actions - self.mu, -1)

            # A = -0.5 (a - mu)P(a - mu)
            self.advantage = -0.5 * -tf.batch_matmul(tf.transpose(action_diff, [0, 2, 1]),
                                                     tf.batch_matmul(self.p_matrix, action_diff))

            with tf.name_scope('q_values'):
                # Q = A + V
                self.q_value = self.v + self.advantage

        # Get all variables under this scope for target network update
        return get_variables(scope)


    def create_training_operations(self):
        """
        NAF update logic.
        """
        with tf.name_scope("update"):
            self.q_targets = tf.placeholder(tf.float32, [None], name='q_targets')

            # MSE
            loss = tf.reduce_mean(tf.squared_difference(self.q_targets, tf.squeeze(self.q_value)), name='loss')
            self.optimize_op = self.optimizer.minimize(loss)


        with tf.name_scope("update_target"):
            # Combine hidden layer variables and output layer variables
            self.training_vars = self.training_model.get_variables() + self.training_output_vars
            self.target_vars = self.target_model.get_variables() + self.target_output_vars

            for v_source, v_target in zip(self.training_vars, self.target_vars):
                update = v_target.assign_sub(self.tau * (v_target - v_source))

                self.target_network_update.append(update)

    def get_target_value_estimate(self, next_states):
        """
        Estimate of next state Q values.
        :param next_states:
        :return:
        """
        return self.session.run(self.v, {self.next_states: next_states})
