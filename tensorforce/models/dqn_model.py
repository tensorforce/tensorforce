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

import tensorflow as tf

from tensorforce.core import Model
from tensorforce.core.networks import NeuralNetwork, layers


class DQNModel(Model):

    default_config = {
        'update_target_weight': 1.0,
        'double_dqn': False
    }
    allows_discrete_actions = True
    allows_continuous_actions = False

    def __init__(self, config, network_builder):
        """
        Training logic for DQN.

        :param config: Configuration dict
        """
        config.default(DQNModel.default_config)
        self.network = network_builder
        super(DQNModel, self).__init__(config)
        self.double_dqn = config.double_dqn

    def create_tf_operations(self, config):
        super(DQNModel, self).create_tf_operations(config)

        # placeholders
        with tf.variable_scope('placeholders'):
            self.q_targets = tf.placeholder(tf.float32, (None,), name='q_targets')

        # training network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(self.network, inputs={name: state[:-1] for name, state in self.state.items()}, episode_length=self.episode_length)
            self.training_internal_states = self.training_network.internal_inits
            self.training_output = layers['linear'](x=self.training_network.output, size=self.num_actions)

        # target network
        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(self.network, inputs={name: state[1:] for name, state in self.state.items()}, episode_length=self.episode_length)
            self.target_internal_states = self.target_network.internal_inits
            self.target_output = layers['linear'](x=self.target_network.output, size=self.num_actions)

        with tf.name_scope('predict'):
            self.dqn_action = tf.argmax(self.training_output, axis=2, name='dqn_action')

        with tf.name_scope('targets'):
            if config.double_dqn:
                selector = tf.one_hot(self.dqn_action, self.num_actions, name='selector')
                self.target_values = tf.reduce_sum(tf.multiply(self.target_output, selector), axis=2, name='target_values')
            else:
                self.target_values = tf.reduce_max(self.target_output, axis=2, name='target_values')

        with tf.name_scope('update'):
            # One_hot tensor of the actions that have been taken
            actions_one_hot = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name='action_one_hot')

            # Training output, so we get the expected rewards given the actual states and actions
            q_values_actions_taken = tf.reduce_sum(self.training_output * actions_one_hot, axis=2,
                                                   name='q_acted')

            # Surrogate loss as the mean squared error between actual observed rewards and expected rewards
            delta = self.q_targets - q_values_actions_taken

            # If gradient clipping is used, calculate the huber loss
            if config.clip_gradients:
                huber_loss = tf.where(tf.abs(delta) < config.clip_value, 0.5 * tf.square(delta), tf.abs(delta) - 0.5)
                loss = tf.reduce_mean(huber_loss, name='compute_surrogate_loss')
            else:
                loss = tf.reduce_mean(tf.square(delta), name='compute_surrogate_loss')
            tf.losses.add_loss(loss)

            # Update target network
            self.target_network_update = []
            with tf.name_scope("update_target"):
                for v_source, v_target in zip(self.training_network.variables, self.target_network.variables):
                    update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                    self.target_network_update.append(update)

    def get_action(self, state, episode=1):
        """
        Returns the predicted action for a given state.

        :param state: State tensor
        :param episode: Current episode
        :return: action number
        """
        fetches = [self.dqn_action]
        fetches.extend(self.training_internal_states)
        fetches.extend(self.target_internal_states)

        feed_dict = {self.state: [(state,)]}
        feed_dict.update({internal_state: self.training_network.internal_inits[n] for n, internal_state in enumerate(self.training_network.internal_state_inputs)})
        feed_dict.update({internal_state: self.target_network.internal_inits[n] for n, internal_state in enumerate(self.target_network.internal_state_inputs)})

        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)

        self.training_internal_states = fetched[1:len(self.training_internal_states)]
        self.target_internal_states = fetched[1 + len(self.training_internal_states):]

        return fetched[0]

    def update(self, batch):
        """
        Perform a single training step and updates the target network.

        :param batch: Mini batch to use for training
        :return: void
        """
        # Compute estimated future value
        float_terminals = batch['terminals'].astype(float)

        if self.double_dqn:
            y = self.session.run(self.target_values, {self.state: [batch['next_states']], self.next_state: [batch['next_states']]})
        else:
            y = self.session.run(self.target_values, {self.next_state: [batch['next_states']]})

        q_targets = batch['rewards'] + (1.0 - float_terminals) * self.discount * y

        feed_dict = {
            self.episode_length: [len(batch['rewards'])],
            self.q_targets: q_targets,
            self.actions: [batch['actions']],
            self.state: [batch['states']]
        }

        fetches = [self.optimize, self.training_output]
        fetches.extend(self.training_network.internal_outputs)
        fetches.extend(self.target_network.internal_outputs)

        for n, internal_state in enumerate(self.training_network.internal_inputs):
            feed_dict[internal_state] = self.training_internal_states[n]

        for n, internal_state in enumerate(self.target_network.internal_inputs):
            feed_dict[internal_state] = self.target_internal_states[n]

        fetched = self.session.run(fetches, feed_dict)

        # Update internal state list, e.g. or LSTM
        self.training_internal_states = fetched[2:len(self.training_internal_states)]
        self.target_internal_states = fetched[2 + len(self.training_internal_states):]

    def update_target_network(self):
        """
        Updates target network.
        :return:
        """
        self.session.run(self.target_network_update)
