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

    default_config = dict(
        update_target_weight=1.0,
        double_dqn=False,
        clip_gradients=0.0
    )
    allows_discrete_actions = True
    allows_continuous_actions = False

    def __init__(self, config):
        """Training logic for DQN.

        Args:
            config: 
            network_builder: 
        """
        config.default(DQNModel.default_config)
        super(DQNModel, self).__init__(config)
        self.double_dqn = config.double_dqn

    def create_tf_operations(self, config):
        super(DQNModel, self).create_tf_operations(config)

        num_actions = {name: action.num_actions for name, action in config.actions}

        # Training network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(config.network, inputs=self.state)

            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)
            training_output = dict()

            for action in self.action:
                training_output[action] = layers['linear'](x=self.training_network.output, size=num_actions[action])
                self.action_taken[action] = tf.argmax(training_output[action], axis=1)

        # Target network
        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(config.network, inputs=self.state)
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)
            target_value = dict()

            for action in self.action:
                target_output = layers['linear'](x=self.target_network.output, size=num_actions[action])
                if config.double_dqn:
                    selector = tf.one_hot(self.action_taken[action], num_actions[action])
                    target_value[action] = tf.reduce_sum(tf.multiply(target_output, selector), axis=1)
                else:
                    target_value[action] = tf.reduce_max(target_output, axis=1)

        with tf.name_scope('update'):
            for action in self.action:
                # One_hot tensor of the actions that have been taken
                action_one_hot = tf.one_hot(self.action[action][:-1], num_actions[action])
                # Training output, so we get the expected rewards given the actual states and actions
                q_value = tf.reduce_sum(training_output[action][:-1] * action_one_hot, axis=1)

                # Surrogate loss as the mean squared error between actual observed rewards and expected rewards
                q_target = self.reward[:-1] + (1.0 - tf.cast(self.terminal[:-1], tf.float32)) * self.discount * target_value[action][1:]
                delta = q_target - q_value

                # If gradient clipping is used, calculate the huber loss
                if config.clip_gradients > 0.0:
                    huber_loss = tf.where(tf.abs(delta) < config.clip_gradients, 0.5 * tf.square(delta), tf.abs(delta) - 0.5)
                    loss = tf.reduce_mean(huber_loss)
                else:
                    loss = tf.reduce_mean(tf.square(delta))
                tf.losses.add_loss(loss)

        # Update target network
        with tf.name_scope("update_target"):
            self.target_network_update = list()
            for v_source, v_target in zip(self.training_network.variables, self.target_network.variables):
                update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                self.target_network_update.append(update)

    def update_target(self):
        """
        Updates target network.
        :return:
        """
        self.session.run(self.target_network_update)
