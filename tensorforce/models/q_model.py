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
Generic q-value-based model with target network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorforce import util
from tensorforce.models import Model
from tensorforce.core.networks import NeuralNetwork


class QModel(Model):

    default_config = dict(
        target_update_frequency=10000,
        update_target_weight=1.0,
        clip_loss=0.0
    )

    def __init__(self, config):
        """Training logic for DQN.

        Args:
            config:
        """
        config.default(QModel.default_config)
        self.last_target_update = 0
        self.target_update_frequency = config.target_update_frequency
        super(QModel, self).__init__(config)

        # Synchronise target with training network
        self.possible_update_target(force=True)

    def create_tf_operations(self, config):
        super(QModel, self).create_tf_operations(config)

        # Placeholders
        with tf.variable_scope('placeholder'):
            self.next_state = dict()
            for name, state in config.states.items():
                self.next_state[name] = tf.placeholder(dtype=util.tf_dtype(state.type), shape=(None,) + tuple(state.shape), name=('next_' + name))

        network_builder = util.get_function(fct=config.network)

        # Training network
        with tf.variable_scope('training') as training_scope:
            self.training_network = NeuralNetwork(network_builder=network_builder, inputs=self.state, summary_level=config.tf_summary_level)
            self.network_internal_index = len(self.internal_inputs)
            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)
            self.q_values = self.create_training_operations(config)
            self.training_variables = tf.contrib.framework.get_variables(scope=training_scope)

        # Target network
        with tf.variable_scope('target') as target_scope:
            self.target_network = NeuralNetwork(network_builder=network_builder, inputs=self.next_state)
            self.next_internal_inputs = list(self.target_network.internal_inputs)
            self.target_values = self.create_target_operations(config)
            self.target_variables = tf.contrib.framework.get_variables(scope=target_scope)

        with tf.name_scope('update'):
            deltas = self.create_q_deltas(config)

            # Surrogate loss as the mean squared error between actual observed rewards and expected rewards
            delta = tf.reduce_mean(input_tensor=tf.concat(values=deltas, axis=1), axis=1)
            self.loss_per_instance = tf.square(delta)

            # If loss clipping is used, calculate the huber loss
            if config.clip_loss > 0.0:
                huber_loss = tf.where(condition=(tf.abs(delta) < config.clip_loss), x=(0.5 * self.loss_per_instance), y=(tf.abs(delta) - 0.5))
                self.q_loss = tf.reduce_mean(input_tensor=huber_loss, axis=0)
            else:
                self.q_loss = tf.reduce_mean(input_tensor=self.loss_per_instance, axis=0)
            tf.losses.add_loss(self.q_loss)

        # for each loss over an action create a summary
        if len(self.q_loss.shape) > 1:
            for action_ind in range(self.q_loss.shape[1]):
                tf.summary.scalar('q-loss-action-{}'.format(action_ind), self.q_loss[action_ind])
        else:
            tf.summary.scalar('q-loss', self.q_loss)

        # Update target network
        with tf.name_scope('update-target'):
            self.target_network_update = list()
            for v_source, v_target in zip(self.training_variables, self.target_variables):
                update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                self.target_network_update.append(update)

    def create_q_deltas(self, config):
        """
        Creates the deltas (or advantage) of the Q values
        :return: A list of deltas per action
        """
        deltas = list()
        terminal_float = tf.cast(x=self.terminal, dtype=tf.float32)
        for name, action in self.action.items():
            reward = self.reward
            terminal = terminal_float
            for _ in range(len(config.actions[name].shape)):
                reward = tf.expand_dims(input=reward, axis=1)
                terminal = tf.expand_dims(input=terminal, axis=1)
            q_target = reward + (1.0 - terminal) * config.discount * self.target_values[name]
            delta = tf.stop_gradient(q_target) - self.q_values[name]
            delta = tf.reshape(tensor=delta, shape=(-1, util.prod(config.actions[name].shape)))
            deltas.append(delta)
        return deltas

    def create_training_operations(self, config):
        """
        Create training network operations. Has to set 'self.action_taken'.
        :return: A dict containing the q-values per action.
        """

    def create_target_operations(self, config):
        """
        Create target network operations.
        :return: A dict containing the target values per action.
        """

    def update_feed_dict(self, batch):
        if 'next_states' in batch:
            # if 'next_states' is given, just use given values
            feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
            feed_dict.update({next_state: batch['next_states'][name] for name, next_state in self.next_state.items()})
            feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})
            feed_dict[self.reward] = batch['rewards']
            feed_dict[self.terminal] = batch['terminals']
            feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})
            feed_dict.update({internal: batch['next_internals'][n] for n, internal in enumerate(self.next_internal_inputs, self.network_internal_index)})
        else:
            # if 'next_states' not explicitly given, assume temporally consistent sequence
            feed_dict = {state: batch['states'][name][:-1] for name, state in self.state.items()}
            feed_dict.update({next_state: batch['states'][name][1:] for name, next_state in self.next_state.items()})
            feed_dict.update({action: batch['actions'][name][:-1] for name, action in self.action.items()})
            feed_dict[self.reward] = batch['rewards'][:-1]
            feed_dict[self.terminal] = batch['terminals'][:-1]
            feed_dict.update({internal: batch['internals'][n][:-1] for n, internal in enumerate(self.internal_inputs)})
            feed_dict.update({internal: batch['internals'][n][1:] for n, internal in enumerate(self.next_internal_inputs, self.network_internal_index)})
        return feed_dict

    def update(self, *args, **kwargs):
        self.possible_update_target()
        return super(QModel, self).update(*args, **kwargs)

    def possible_update_target(self, force=False):
        """
        Updates target network if necessary
        :return:
        """
        if self.timestep > self.last_target_update + self.target_update_frequency or force:
            self.last_target_update = self.timestep
            self.session.run(self.target_network_update)
