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
from tensorforce.neural_networks import NeuralNetwork
from tensorforce.util.experiment_util import global_seed
from tensorforce.updater import Model
from tensorforce.util.exploration_util import exploration_mode


class DeepQNetwork(Model):
    default_config = {
        'double_dqn': False,
        'tau': 0.5,
        'epsilon': 1.0,
        'epsilon_final': 0.1,
        'epsilon_states': 1e6,
        'gamma': 0.99,
        'alpha': 0.00025,
        'clip_gradients': True,
        'clip_value': 1.0
    }

    def __init__(self, config):
        """
        Training logic for DQN.

        :param config: Configuration parameters
        """
        super(DeepQNetwork, self).__init__(config)

        self.config = create_config(config, default=self.default_config)
        self.action_count = self.config.actions
        self.tau = self.config.tau
        self.epsilon = self.config.epsilon
        self.epsilon_final = self.config.epsilon_final
        self.epsilon_states = self.config.epsilon_states
        self.gamma = self.config.gamma
        self.alpha = self.config.alpha
        self.batch_size = self.config.batch_size

        self.double_dqn = self.config.double_dqn

        self.gradient_clipping = None
        if self.config.clip_gradients:
            self.gradient_clipping = self.config.clip_value

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.exploration = exploration_mode['epsilon_decay']

        # Input placeholders
        self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
        self.next_states = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="next_states")
        self.terminals = tf.placeholder(tf.float32, self.batch_shape, name='terminals')
        self.rewards = tf.placeholder(tf.float32, self.batch_shape, name='rewards')

        self.target_network_update = []

        # output layer
        output_layer_config = [{
            "type": "dense",
            "neurons": self.config.actions,
            "weight_init": "tensorflow.contrib.layers.python.layers.initializers.xavier_initializer",
            "weight_init_param": {},
            "bias_init": "tensorflow.python.ops.init_ops.zeros_initializer",
            "bias_init_param": None,
            "activation": "tensorflow.python.ops.nn.relu",
            "activation_param": None,
            "regularization": None,
            "regularization_param": None
        }]

        scope = '' if self.config.tf_scope is None else self.config.tf_scope + '-'
        self.training_model = NeuralNetwork(self.config.network_layers + output_layer_config, self.state, scope=scope + 'training')
        self.target_model = NeuralNetwork(self.config.network_layers + output_layer_config, self.next_states, scope=scope + 'target')
        self.training_output = self.training_model.get_output()
        self.target_output = self.target_model.get_output()

        # Create training operations
        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.create_training_operations()
        self.saver = tf.train.Saver()
        writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())
        self.session.run(tf.global_variables_initializer())

    def get_action(self, state, episode=1, total_states=0):
        """
        Returns the predicted action for a given state.

        :param state: State tensor
        :param episode: Current episode
        :return:
        """

        epsilon = self.exploration(self.epsilon_final, total_states, self.epsilon_states, self.epsilon)

        if self.random.random_sample() < epsilon:
            action = self.random.randint(0, self.action_count)
        else:
            if self.double_dqn:
                action = self.session.run(self.dqn_action, {self.state: [state]})[0]
            else:
                action = self.session.run(self.dqn_action, {self.state: [state]})
        return action

    def update(self, batch):
        """
        Perform a single training step and updates the target network.

        :param batch: Mini batch to use for training
        :return:
        """

        # Compute estimated future value
        float_terminals = batch['terminals'].astype(float)
        q_targets = batch['rewards'] + (1. - float_terminals) \
                                       * self.gamma * self.get_target_values(batch['next_states'])

        self.session.run([self.optimize_op, self.training_output, self.target_network_update], {
            self.q_targets: q_targets,
            self.actions: batch['actions'],
            self.state: batch['states']})

    def create_training_operations(self):
        """
        Create graph operations for loss computation and
        target network updates.

        :return:
        """

        with tf.name_scope("predict"):
            self.dqn_action = tf.argmax(self.training_output, dimension=1, name='dqn_action')

        with tf.name_scope("targets"):
            if self.double_dqn:
                selector = tf.one_hot(self.dqn_action, self.action_count, name='selector')
                self.target_values = tf.reduce_sum(tf.multiply(self.target_output, selector), reduction_indices=1,
                                                   name='target_values')
            else:
                self.target_values = tf.reduce_max(self.target_output, reduction_indices=1,
                                                   name='target_values')

        with tf.name_scope("update"):
            self.q_targets = tf.placeholder(tf.float32, [None], name='q_targets')
            self.actions = tf.placeholder(tf.int32, [None], name='actions')

            # Q values for actions taken in batch
            actions_one_hot = tf.one_hot(self.actions, self.action_count, 1.0, 0.0, name='action_one_hot')
            q_values_actions_taken = tf.reduce_sum(self.training_output * actions_one_hot, reduction_indices=1,
                                                   name='q_acted')

            # Mean squared error
            loss = tf.reduce_mean(tf.square(self.q_targets - q_values_actions_taken), name='loss')

            if self.gradient_clipping is not None:
                grads_and_vars = self.optimizer.compute_gradients(loss)
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grads_and_vars[idx] = (tf.clip_by_norm(grad, self.gradient_clipping), var)
                self.optimize_op = self.optimizer.apply_gradients(grads_and_vars)

            self.optimize_op = self.optimizer.minimize(loss)

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
