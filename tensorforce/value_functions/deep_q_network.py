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
Deep Q network. Implements training and update logic as described
in the DQN paper.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import tensorflow as tf

from tensorforce.config import create_config
from tensorforce.neural_networks.neural_network import get_network
from tensorforce.util.experiment_util import global_seed
from tensorforce.value_functions.value_function import ValueFunction


class DeepQNetwork(ValueFunction):

    default_config = {
        'tau': 0,
        'epsilon': 0.1,
        'gamma': 0,
        'alpha': 0.5,
        'clip_gradients': False
    }

    def __init__(self, config):
        """
        Training logic for DQN.

        :param config: Configuration parameters
        """
        super(DeepQNetwork, self).__init__(config)

        self.config = create_config(config, default=self.default_config)
        self.actions = self.config.actions
        self.tau = self.config.tau
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.alpha = self.config.alpha

        if self.config.clip_gradients:
            self.gradient_clipping = self.config.clip_value

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        # Input placeholders
        self.state = tf.placeholder(tf.float32, [None, self.config.state_shape], name="state")
        self.next_states = tf.placeholder(tf.float32, [None, self.config.state_shape], name="next_states")
        self.actions = tf.placeholder(tf.int64, [None], name='actions')
        self.terminals = tf.placeholder(tf.float32, [None], name='terminals')
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
        self.target_network_update = []

        self.training_network = get_network(self.config.network_layers, self.state, 'training')
        self.target_network = get_network(self.config.network_layers, self.next_states, 'target')

        # Create training operations
        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.create_training_operations()

        self.session.run(tf.initialize_all_variables())

    def get_action(self, state):
        """
        Returns the predicted action for a given state.

        :param state: State tensor
        :return:
        """

        if self.random.random_sample() < self.epsilon:
            return self.random.randint(0, self.actions)
        else:
            # TODO partial run here?
            return self.session.run(self.dqn_action, {self.state: state})[0]

    def update(self, batch):
        """
        Perform a single training step and updates the target network.

        :param batch: Mini batch to use for training
        :return:
        """
        # Use y values to compute loss and update
        self.session.run([self.optimize_op, self.target_network_update], {
                             self.state: batch['states'],
                             self.rewards: batch['rewards'],
                             self.actions: batch['actions'],
                             self.next_states: batch['next_states'],
                             self.terminals: batch['terminals']})

    def create_training_operations(self):
        """
        Create graph operations for loss computation and
        target network updates.

        :return:
        """

        with tf.name_scope("predict"):
            self.dqn_action = tf.argmax(self.training_network, dimension=1, name='dqn_action')

        with tf.name_scope("training"):
            float_terminals = np.array(self.terminals, dtype=float)

            target_values = tf.reduce_max(self.target_network, reduction_indices=1,
                                          name='target_values')

            q_targets = self.rewards + (1. - float_terminals) * self.gamma * target_values

            actions_one_hot = tf.one_hot(self.actions, self.actions, 1.0, 0.0)

            batch_q_values = tf.identity(self.training_network, name="batch_q_values")
            q_values_actions_taken = tf.reduce_sum(batch_q_values * actions_one_hot, reduction_indices=1,
                                                   name='q_acted')

            # Mean squared error
            loss = tf.reduce_mean(tf.square(q_targets - q_values_actions_taken), name='loss')

            if self.gradient_clipping is not None:
                grads_and_vars = self.optimizer.compute_gradients(loss)

                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grads_and_vars[idx] = (tf.clip_by_norm(grad, self.gradient_clipping), var)
                self.optimize_op = self.optimizer.apply_gradients(grads_and_vars)
            else:
                self.optimize_op = self.optimizer.apply_gradients(loss)

        # Update target network with update weight tau
        with tf.name_scope("update_target"):
            for v_source, v_target in zip(self.training_network.variables(), self.training_network.variables()):
                operation = v_target.assign_sub(self.tau * (v_target - v_source))

                self.target_network_update.append(operation)
