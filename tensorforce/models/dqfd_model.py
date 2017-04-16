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
Model for deep-q learning from demonstration. Principal structure similar to (double) deep-q-networks
but uses additional loss terms.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorforce.models import Model
from tensorforce.models.neural_networks import NeuralNetwork
from tensorforce.util.experiment_util import global_seed


# UNDER CONSTRUCTION
class DQFDMOdel(Model):

    def __init__(self, config, scope, define_network=None):
        super(DQFDMOdel, self).__init__(config, scope)

        self.action_count = self.config.actions
        self.tau = self.config.tau
        self.gamma = self.config.gamma

        self.clip_value = None
        if self.config.clip_gradients:
            self.clip_value = self.config.clip_value

        if self.config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.target_network_update = []

        # Output layer
        output_layer_config = [{"type": "linear", "num_outputs": self.config.actions, "trainable": True}]

        # Input placeholders
        self.state_shape = tuple(self.config.state_shape)
        self.state = tf.placeholder(tf.float32, (None, None) + self.state_shape, name="state")
        self.next_states = tf.placeholder(tf.float32, (None, None) + self.state_shape,
                                          name="next_states")
        self.terminals = tf.placeholder(tf.float32, (None, None), name='terminals')
        self.rewards = tf.placeholder(tf.float32, (None, None), name='rewards')

        if define_network is None:
            define_network = NeuralNetwork.layered_network(self.config.network_layers + output_layer_config)

        self.training_network = NeuralNetwork(define_network, [self.state], episode_length=self.episode_length,
                                              scope=self.scope + 'training')
        self.target_network = NeuralNetwork(define_network, [self.next_states], episode_length=self.episode_length,
                                            scope=self.scope + 'target')

        self.training_internal_states = self.training_network.internal_state_inits
        self.target_internal_states = self.target_network.internal_state_inits

        self.training_output = self.training_network.output
        self.target_output = self.target_network.output

        # Create training operations
        self.create_training_operations()

        self.init_op = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
        self.session.run(self.init_op)

    def pretrain_update(self, batch):
        """
        Computes the pre-training update.
        :param batch: 
        :return: 
        """
        self.logger.debug('Computing pre-training update..')
        pass

    def update(self, batch):
        pass

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
            # N.b. this is not necessary for original dqfd, but we want to allow recurrent q-networks too
            fetches = [self.dqn_action]
            fetches.extend(self.training_internal_states)
            fetches.extend(self.target_internal_states)

            feed_dict = {self.episode_length: [1], self.state: [(state,)]}

            feed_dict.update({internal_state: self.training_network.internal_state_inits[n] for n, internal_state in
                              enumerate(self.training_network.internal_state_inputs)})
            feed_dict.update({internal_state: self.target_network.internal_state_inits[n] for n, internal_state in
                              enumerate(self.target_network.internal_state_inputs)})

            fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)
            # First element of output list is action
            action = fetched[0][0]

            # Update optional internal states, e.g. LSTM cells
            self.training_internal_states = fetched[1:1 + len(self.training_internal_states)]
            self.target_internal_states = fetched[1 + len(self.training_internal_states):]

        self.total_states += 1

        return action

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

    def create_training_operations(self):
        """
        Create training graph. For DQFD, we build the double-dqn training graph and
        modify the double_q_loss function according to eq. 5
        
        """
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

            # One_hot tensor of the actions that have been taken
            actions_one_hot = tf.one_hot(self.actions, self.action_count, 1.0, 0.0, name='action_one_hot')

            # Training output, so we get the expected rewards given the actual states and actions
            q_values_actions_taken = tf.reduce_sum(self.training_output * actions_one_hot, axis=2,
                                                   name='q_acted')

            delta = self.q_targets - q_values_actions_taken

            #TODO introduce other loss components
            self.double_q_loss = tf.reduce_mean(tf.square(delta), name='compute_surrogate_loss')

            self.grads_and_vars = self.optimizer.compute_gradients(self.double_q_loss)
            self.optimize_op = self.optimizer.apply_gradients(self.grads_and_vars)

            # Update target network with update weight tau
            with tf.name_scope("update_target"):
                for v_source, v_target in zip(self.training_network.variables, self.target_network.variables):
                    update = v_target.assign_sub(self.tau * (v_target - v_source))
                    self.target_network_update.append(update)

