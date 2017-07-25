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

from tensorforce import util
from tensorforce.models import Model
from tensorforce.core.networks import NeuralNetwork, layers


class DQNModel(Model):

    allows_discrete_actions = True
    allows_continuous_actions = False

    default_config = dict(
        update_target_weight=1.0,
        double_dqn=False,
        clip_loss=0.0
    )

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

        flat_action_sizes = {name: util.prod(action.shape) * action.num_actions for name, action in config.actions}
        action_shapes = {name: (-1,) + action.shape + (action.num_actions,) for name, action in config.actions}

        # Training network
        with tf.variable_scope('training'):
            network_builder = util.get_function(fct=config.network)
            self.training_network = NeuralNetwork(network_builder=network_builder, inputs=self.state)
            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)

            self.training_output = dict()
            for action in self.action:
                output = layers['linear'](x=self.training_network.output, size=flat_action_sizes[action])
                self.training_output[action] = tf.reshape(tensor=output, shape=action_shapes[action])
                self.action_taken[action] = tf.argmax(self.training_output[action], axis=-1)

        # Target network
        with tf.variable_scope('target'):
            network_builder = util.get_function(fct=config.network)
            self.target_network = NeuralNetwork(network_builder=network_builder, inputs=self.state)
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)

            target_value = dict()
            for action in self.action:
                output = layers['linear'](x=self.target_network.output, size=flat_action_sizes[action])
                output = tf.reshape(tensor=output, shape=action_shapes[action])
                if config.double_dqn:
                    selector = tf.one_hot(indices=self.action_taken[action], depth=action_shapes[action][1])
                    target_value[action] = tf.reduce_sum(input_tensor=(output * selector), axis=-1)
                else:
                    target_value[action] = tf.reduce_max(input_tensor=output, axis=-1)

        with tf.name_scope('update'):
            self.actions_one_hot = dict()
            self.q_values = dict()
            deltas = list()
            for action in self.action:
                # One_hot tensor of the actions that have been taken
                self.actions_one_hot[action] = tf.one_hot(indices=self.action[action][:-1], depth=config.actions[action].num_actions)

                # Training output, so we get the expected rewards given the actual states and actions
                self.q_values[action] = tf.reduce_sum(input_tensor=(self.training_output[action][:-1] * self.actions_one_hot[action]), axis=-1)

                reward = self.reward[:-1]
                terminal = tf.cast(x=self.terminal[:-1], dtype=tf.float32)
                for _ in range(len(config.actions[action].shape)):
                    reward = tf.expand_dims(input=reward, axis=1)
                    terminal = tf.expand_dims(input=terminal, axis=1)

                # Surrogate loss as the mean squared error between actual observed rewards and expected rewards
                q_target = reward + (1.0 - terminal) * config.discount * target_value[action][1:]
                delta = q_target - self.q_values[action]

                ds_list = [delta]
                for _ in range(len(config.actions[action].shape)):
                    ds_list = [d for ds in ds_list for d in tf.unstack(value=ds, axis=1)]
                deltas.extend(ds_list)

            delta = tf.add_n(inputs=deltas) / len(deltas)
            self.loss_per_instance = tf.square(delta)

            # If gradient clipping is used, calculate the huber loss
            if config.clip_loss > 0.0:
                huber_loss = tf.where(condition=(tf.abs(delta) < config.clip_gradients), x=(0.5 * self.loss_per_instance), y=(tf.abs(delta) - 0.5))
                loss = tf.reduce_mean(input_tensor=huber_loss, axis=0)
            else:
                loss = tf.reduce_mean(input_tensor=self.loss_per_instance, axis=0)
            self.dqn_loss = loss
            tf.losses.add_loss(loss)

        # Update target network
        with tf.name_scope('update_target'):
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
