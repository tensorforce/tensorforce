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
Model for deep-q learning from demonstration. Principal structure similar to double deep-q-networks
but uses additional loss terms for demo data.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorforce.core.model import Model
from tensorforce.core.networks import NeuralNetwork, layers


class DQFDModel(Model):

    default_config = {
        'update_target_weight': 1.0,
        'double_dqn': False
    }
    allows_discrete_actions = True
    allows_continuous_actions = False

    def __init__(self, config, network_builder):
        super(DQFDModel, self).__init__(config)
        self.network = network_builder
        config.default(DQFDModel.default_config)

        self.double_dqn = config.double_dqn
        self.supervised_weight = config.supervised_weight
        self.expert_margin = config.expert_margin

    def pre_train_update(self, batch):
        """
        Computes the pre-training update.
        
        :param batch: Demo batch data
        :return: 
        """

        self.logger.debug('Computing pre-training update..')

        # Compute estimated future value
        float_terminals = batch['terminals'].astype(float)
        y = self.get_target_values(batch['next_states'])

        q_targets = batch['rewards'] + (1. - float_terminals) \
                                       * self.discount * y

        feed_dict = {
            self.episode_length: [len(batch['rewards'])],
            self.q_targets: q_targets,
            self.actions: [batch['actions']],
            self.expert_actions: [batch['actions']],  # Separate placeholders -> separate loss components
            self.state: [batch['states']]
        }

        fetches = [self.optimize_dqfd, self.training_output]

        # Internal state management for recurrent dqn
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

    def update(self, online_batch, demo_batch=None):
        """
        Updates by applying the dqfd loss on the demo data and the double-Q
        loss on the online data.
                
        :param online_batch: 
        :param demo_batch: 

        """
        # Delegate demo batch update to pretraining method
        self.pre_train_update(demo_batch)

        self.logger.debug('Computing online update..')

        # Compute estimated future value
        float_terminals = online_batch['terminals'].astype(float)
        y = self.get_target_values(online_batch['next_states'])

        q_targets = online_batch['rewards'] + (1. - float_terminals) \
                                       * self.discount * y

        feed_dict = {
            self.episode_length: [len(online_batch['rewards'])],
            self.q_targets: q_targets,
            self.actions: [online_batch['actions']],

            self.state: [online_batch['states']]
        }

        fetches = [self.optimize_double_q, self.training_output]

        # Internal state management for recurrent dqn
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

    def get_target_values(self, next_states):
        """
        Estimate of next state Q values using double q estimate.
        
        :param next_states:
        """

        return self.session.run(self.target_values, {self.state: [next_states], self.next_states: [next_states]})

    def update_target_network(self):
        """
        Updates target network.

        """
        self.session.run(self.target_network_update)

    def create_tf_operations(self, config):
        """
        Create training graph. For DQFD, we build the double-dqn training graph and
        modify the double_q_loss function according to eq. 5
        
        """
        super(DQFDModel, self).create_tf_operations(config)

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

        with tf.name_scope("predict"):
            self.dqn_action = tf.argmax(self.training_output, axis=2, name='dqn_action')

        with tf.name_scope("targets"):
            selector = tf.one_hot(self.dqn_action, self.action_count, name='selector')
            self.target_values = tf.reduce_sum(tf.multiply(self.target_output, selector), axis=2,
                                               name='target_values')

        with tf.name_scope("update"):
            # Self.q_targets gets fed the actual observed rewards and expected future rewards
            self.q_targets = tf.placeholder(tf.float32, (None, None), name='q_targets')

            # Self.actions gets fed the actual actions that have been taken
            self.actions = tf.placeholder(tf.int32, (None, None), name='actions')
            self.expert_actions = tf.placeholder(tf.int32, (None, None), name='expert_actions')

            # One_hot tensor of the actions that have been taken
            actions_one_hot = tf.one_hot(self.actions, self.action_count, 1.0, 0.0, name='action_one_hot')

            # Training output, so we get the expected rewards given the actual states and actions
            q_values_actions_taken = tf.reduce_sum(self.training_output * actions_one_hot, axis=2,
                                                   name='q_acted')

            # Expert action Q values
            expert_actions_one_hot = tf.one_hot(self.expert_actions, self.action_count, 1.0, 0.0, name='action_one_hot')
            q_values_expert_actions = tf.reduce_sum(self.training_output * expert_actions_one_hot, axis=2,
                                                   name='q_expert_acted')

            delta = self.q_targets - q_values_actions_taken

            self.double_q_loss = tf.reduce_mean(tf.square(delta), name='compute_surrogate_loss')

            # Create the supervised margin loss
            mask = tf.ones_like(expert_actions_one_hot, dtype=tf.float32)

            # Zero for the action taken, one for all other actions, now multiply by expert margin
            inverted_one_hot = mask - expert_actions_one_hot

            # max_a([Q(s,a) + l(s,a_E,a)], l(s,a_E, a) is 0 for expert action and margin value for others
            expert_margin = self.training_output + tf.multiply(inverted_one_hot, self.expert_margin)

            supervised_selector = tf.reduce_max(expert_margin, axis=2, name='expert_margin_selector')

            # J_E(Q) = max_a([Q(s,a) + l(s,a_E,a)] - Q(s,a_E)
            self.supervised_loss = supervised_selector - q_values_expert_actions

            # Combining double q loss with supervised loss
            self.dqfd_loss = self.double_q_loss + self.supervised_weight * self.supervised_loss

            # This decomposition is not necessary, we just want to be able to export gradients
            self.double_q_grads_and_vars = self.optimizer.compute_gradients(self.double_q_loss)
            self.dqfd_grads_and_vars = self.optimizer.compute_gradients(self.dqfd_loss)

            self.optimize_double_q = self.optimizer.apply_gradients(self.double_q_grads_and_vars)
            self.optimize_dqfd = self.optimizer.apply_gradients(self.dqfd_grads_and_vars)

            # Update target network with update weight tau
            self.target_network_update = []
            with tf.name_scope("update_target"):
                for v_source, v_target in zip(self.training_network.variables, self.target_network.variables):
                    update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                    self.target_network_update.append(update)

