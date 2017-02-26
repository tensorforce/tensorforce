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
Deep Q network. Implements training and update logic as described
in the DQN paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorforce.models import Model
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.util.experiment_util import global_seed

from tensorforce.default_configs import DQNModelConfig


class DQNModel(Model):
    default_config = DQNModelConfig

    def __init__(self, config, scope):
        """
        Training logic for DQN.

        :param config: Configuration dict
        """
        super(DQNModel, self).__init__(config, scope)

        self.action_count = self.config.actions
        self.tau = self.config.tau
        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size

        self.double_dqn = self.config.double_dqn

        self.clip_value = None
        if self.config.clip_gradients:
            self.clip_value = self.config.clip_value

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.target_network_update = []

        # output layer
        output_layer_config = [{"type": "linear", "num_outputs": self.config.actions, "trainable": True}]

        self.device = self.config.tf_device
        if self.device == 'replica':
            self.device = tf.train.replica_device_setter(ps_tasks=1, worker_device=self.config.tf_worker_device)

        with tf.device(self.device):
            # Input placeholders
            self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
            self.next_states = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape),
                                              name="next_states")
            self.terminals = tf.placeholder(tf.float32, self.batch_shape, name='terminals')
            self.rewards = tf.placeholder(tf.float32, self.batch_shape, name='rewards')

            self.training_model = NeuralNetwork(self.config.network_layers + output_layer_config, self.state,
                                                scope=self.scope + 'training')
            self.target_model = NeuralNetwork(self.config.network_layers + output_layer_config, self.next_states,
                                              scope=self.scope + 'target')

            self.training_output = self.training_model.get_output()
            self.target_output = self.target_model.get_output()

            # Create training operations
            self.create_training_operations()
            self.optimizer = tf.train.RMSPropOptimizer(self.alpha, momentum=0.95, epsilon=0.01)

        self.training_output = self.training_model.get_output()
        self.target_output = self.target_model.get_output()

        self.init_op = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

    def initialize(self):
        self.session.run(self.init_op)

    def get_action(self, state, episode=1):
        """
        Returns the predicted action for a given state.

        :param state: State tensor
        :param episode: Current episode
        :return: action number
        """

        epsilon = self.exploration(episode, self.total_states)

        if self.random.random_sample() < epsilon:
            action = self.random.randint(0, self.action_count)
        else:
            if self.double_dqn:
                action = self.session.run(self.dqn_action, {self.state: [state]})[0]
            else:
                action = self.session.run(self.dqn_action, {self.state: [state]})

        self.total_states += 1
        return action

    def update(self, batch):
        """
        Perform a single training step and updates the target network.

        :param batch: Mini batch to use for training
        :return: void
        """

        # Compute estimated future value
        float_terminals = batch['terminals'].astype(float)
        q_targets = batch['rewards'] + (1. - float_terminals) \
                                       * self.gamma * self.get_target_values(batch['next_states'])

        self.session.run([self.optimize_op, self.training_output], {
            self.q_targets: q_targets,
            self.actions: batch['actions'],
            self.state: batch['states']})

    def get_variables(self):
        return self.training_model.get_variables()

    def assign_variables(self, values):
        assign_variables_ops = [variable.assign(value) for variable, value in zip(self.get_variables(), values)]
        self.session.run(tf.group(assign_variables_ops))

    def get_gradients(self):
        return self.grads_and_vars

    def apply_gradients(self, grads_and_vars):
        apply_gradients_op = self.optimizer.apply_gradients(grads_and_vars)
        self.session.run(apply_gradients_op)

    def create_training_operations(self):
        """
        Create graph operations for compute_surrogate_loss computation and
        target network updates.

        :return:
        """
        with tf.name_scope(self.scope):
            with tf.name_scope("predict"):
                self.dqn_action = tf.argmax(self.training_output, axis=1, name='dqn_action')

            with tf.name_scope("targets"):
                if self.double_dqn:
                    selector = tf.one_hot(self.dqn_action, self.action_count, name='selector')
                    self.target_values = tf.reduce_sum(tf.multiply(self.target_output, selector), axis=1,
                                                       name='target_values')
                else:
                    self.target_values = tf.reduce_max(self.target_output, axis=1,
                                                       name='target_values')

            with tf.name_scope("update"):
                # Self.q_targets gets fed the actual observed rewards and expected future rewards
                self.q_targets = tf.placeholder(tf.float32, [None], name='q_targets')

                # Self.actions gets fed the actual actions that have been taken
                self.actions = tf.placeholder(tf.int32, [None], name='actions')

                # One_hot tensor of the actions that have been taken
                actions_one_hot = tf.one_hot(self.actions, self.action_count, 1.0, 0.0, name='action_one_hot')

                # Training output, so we get the expected rewards given the actual states and actions
                q_values_actions_taken = tf.reduce_sum(self.training_output * actions_one_hot, axis=1,
                                                       name='q_acted')

                # Surrogate loss as the mean squared error between actual observed rewards and expected rewards
                delta = self.q_targets - q_values_actions_taken

                # if gradient clipping is used, calculate the huber loss
                if self.config.clip_gradients:
                    huber_loss = tf.where(tf.abs(delta) < self.clip_value, 0.5 * tf.square(delta), tf.abs(delta) - 0.5)
                    self.loss = tf.reduce_mean(huber_loss, name='compute_surrogate_loss')
                else:
                    self.loss = tf.reduce_mean(tf.square(delta), name='compute_surrogate_loss')

                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.optimize_op = self.optimizer.apply_gradients(self.grads_and_vars)

            # Update target network with update weight tau
            with tf.name_scope("update_target"):
                for v_source, v_target in zip(self.training_model.get_variables(), self.target_model.get_variables()):
                    update = v_target.assign_sub(self.tau * (v_target - v_source))
                    self.target_network_update.append(update)

    def get_target_values(self, next_states):
        """
        Estimate of next state Q values.
        :param next_states:
        :return:
        """
        if self.double_dqn:
            return self.session.run(self.target_values, {self.state: next_states, self.next_states: next_states})
        else:
            return self.session.run(self.target_values, {self.next_states: next_states})

    def update_target_network(self):
        """
        Updates target network.

        :return:
        """
        self.session.run(self.target_network_update)
