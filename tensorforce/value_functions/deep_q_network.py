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

import tensorflow as tf

from tensorforce.neural_networks.neural_network import get_network
from tensorforce.util.experiment_util import global_seed
import numpy as np


class DeepQNetwork(object):
    def __init__(self, agent_config, network_config, deterministic_mode=False):

        # TODO session/executioner config
        self.target_network_update = []
        self.session = tf.Session()
        self.agent_config = agent_config
        self.training_network = get_network(network_config, 'training')
        self.target_network = get_network(network_config, 'target')

        # TODO which config does this belong in? the separation of configs is somewhat artificial
        self.tau = network_config['tau']
        self.actions = network_config['actions']
        self.epsilon = network_config['epsilon']
        self.gamma = network_config['gamma']
        self.alpha = network_config['alpha']

        if agent_config['clip_gradients']:
            self.gradient_clipping = agent_config['clip_value']

        if deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        # Input placeholder
        self.state = tf.placeholder(tf.float32, None, name="state")
        self.batch_states = tf.placeholder(tf.float32, None, name="batch_states")
        self.next_states = tf.placeholder(tf.float32, None, name="next_states")

        # Forward Variables
        self.batch_q_values = tf.identity(self.training_network(self.batch_states), name="batch_q_values")
        self.dqn_action = tf.argmax(self.training_network(self.state), dimension=1, name='dqn_action')

        # Predicted max value of target network
        self.target_values = tf.reduce_max(self.target_network, reduction_indices=1, name='target_values')

        # Create training operations
        self.create_training_operations()

        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.session.run(tf.initialize_all_variables())

    def get_action(self, state):
        """
        Returns the predicted action for a given state.
        :param state: State tensor
        :return:
        """

        if self.random.random_sample() < self.epsilon:
            return self.random.randint(0, self.actions - 1)
        else:
            # TODO partial run here?
            return self.session.run(self.dqn_action, {self.state: state})[0]

    def update(self, batch):
        """
        Perform a single training step and updates the target network.

        :param batch: Mini batch to use for training
        :return:
        """
        float_terminals = np.array(batch['terminals'], dtype=float)
        q_targets = self.session.run(self.target_values, {self.next_states: batch['next_states']})
        y = batch['rewards'] + (1. - float_terminals) * self.gamma * q_targets

        # Use y values to compute loss and update
        self.session.run([self.optimize_op, self.training_network, self.loss], {
            self.q_targets: y,
            self.batch_actions: batch['actions'],
            self.batch_states: batch['states']})

        # Update target network
        self.session.run(self.target_network_update)

    def create_training_operations(self):
        """
        Create graph operations for loss computation and
        target network updates.
        :return:
        """

        with tf.name_scope("training"):
            self.q_targets = tf.placeholder('float32', [None], name='batch_q_targets')
            self.batch_actions = tf.placeholder('int64', [None], name='batch_actions')

            actions_one_hot = tf.one_hot(self.batch_actions, self.actions, 1.0, 0.0, name='one_hot')

            # Not sure if slim's network works like this
            q_values_actions_taken = tf.reduce_sum(self.batch_q_values * actions_one_hot, reduction_indices=1,
                                                   name='q_acted')

            # Mean squared error
            self.loss = tf.reduce_mean(tf.square(self.q_targets - q_values_actions_taken), name='loss')

            if self.gradient_clipping is not None:
                grads_and_vars = self.optimizer.compute_gradients(self.loss)
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grads_and_vars[idx] = (tf.clip_by_norm(grad, self.gradient_clipping), var)
                self.optimize_op = self.optimizer.apply_gradients(grads_and_vars)
            else:
                self.optimize_op = self.optimizer.apply_gradients(self.loss)

        # Update target network with update weight tau
        with tf.name_scope("update_target"):
            for v_source, v_target in zip(self.training_network.variables(), self.training_network.variables()):
                operation = v_target.assign_sub(self.tau * (v_target - v_source))

                self.target_network_update.append(operation)
