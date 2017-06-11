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
        'clip_gradients': 0.0,
        "supervised_weight": 1.0,
        "expert_margin": 0.8
    }

    allows_discrete_actions = True
    allows_continuous_actions = False

    def __init__(self, config):
        config.default(DQFDModel.default_config)
        super(DQFDModel, self).__init__(config)

    def pre_train_update(self, batch=None):
        """Computes the pre-training update.

        Args:
            batch: A batch of demo data.

        Returns:

        """

        fetches = self.dqfd_opt

        feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
        feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})

        feed_dict[self.reward] = batch['rewards']
        feed_dict[self.terminal] = batch['terminals']
        feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

        self.session.run(fetches=fetches, feed_dict=feed_dict)

    def update_target_network(self):
        """
        Updates target network.

        """
        self.session.run(self.target_network_update)

    def create_tf_operations(self, config):
        """Create training graph. For DQFD, we build the double-dqn training graph and
        modify the double_q_loss function according to eq. 5
        
        Args:
            config: Config dict.

        Returns:

        """
        super(DQFDModel, self).create_tf_operations(config)

        num_actions = {name: action.num_actions for name, action in config.actions}

        # placeholders
        with tf.variable_scope('placeholders'):
            self.q_targets = tf.placeholder(tf.float32, (None,), name='q_targets')

        # Training network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(config.network, inputs={name: state for name, state in self.state.items()})
            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)

            training_output = dict()

            for action in self.action:
                training_output[action] = layers['linear'](x=self.training_network.output, size=num_actions[action])
                self.action_taken[action] = tf.argmax(training_output[action], axis=1)

        # Target network
        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(config.network, inputs={name: state for name, state in self.state.items()})
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)

            target_value = dict()

            for action in self.action:
                target_output = layers['linear'](x=self.target_network.output, size=num_actions[action])
                selector = tf.one_hot(self.action_taken[action], num_actions[action])
                target_value[action] = tf.reduce_sum(tf.multiply(target_output, selector), axis=1)

        with tf.name_scope("update"):
            self.dqfd_opt = []

            for action in self.action:
                # Self.q_targets gets fed the actual observed rewards and expected future rewards
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
                    double_q_loss = tf.reduce_mean(huber_loss)
                else:
                    double_q_loss = tf.reduce_mean(tf.square(delta))

                # Use the existing loss structure from the model here, then compute dqfd loss separately
                tf.losses.add_loss(double_q_loss)

                # Create the supervised margin loss
                mask = tf.ones_like(action_one_hot, dtype=tf.float32)

                # Zero for the action taken, one for all other actions, now multiply by expert margin
                inverted_one_hot = mask - action_one_hot

                # max_a([Q(s,a) + l(s,a_E,a)], l(s,a_E, a) is 0 for expert action and margin value for others
                expert_margin = training_output[action][:-1] + tf.multiply(inverted_one_hot, config.expert_margin)

                supervised_selector = tf.reduce_max(expert_margin, axis=1, name='expert_margin_selector')

                # J_E(Q) = max_a([Q(s,a) + l(s,a_E,a)] - Q(s,a_E)
                supervised_loss = supervised_selector - q_value

                # Combining double q loss with supervised loss
                dqfd_loss = double_q_loss + tf.multiply(tf.reduce_mean(supervised_loss), config.supervised_weight)

                # This decomposition is not necessary, we just want to be able to export gradients
                dqfd_grads_and_vars = self.optimizer.compute_gradients(dqfd_loss)

                self.dqfd_opt.append(self.optimizer.apply_gradients(dqfd_grads_and_vars))

        # Update target network according to update weight
        self.target_network_update = []

        with tf.name_scope("update_target"):
            for v_source, v_target in zip(self.training_network.variables, self.target_network.variables):
                update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                self.target_network_update.append(update)

