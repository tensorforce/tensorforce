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
Categorical Deep Q network. Implements training and update logic as described
in the paper A Distributional Perspective on Reinforcement Learning. https://arxiv.org/abs/1707.06887
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorforce import util
from tensorforce.models import Model
from tensorforce.core.networks import NeuralNetwork, layers


class CategoricalDQNModel(Model):

    allows_discrete_actions = True
    allows_continuous_actions = False

    default_config = dict(
        update_target_weight=1.0,
        distribution_max=10,
        distribution_min=-10,
        num_atoms=51,
    )

    def __init__(self, config):
        """Training logic for Categorical DQN.

        Args:
            config:
            network_builder:
        """
        config.default(CategoricalDQNModel.default_config)
        self.distribution_max = config.distribution_max
        self.distribution_min = config.distribution_min
        self.num_atoms = config.num_atoms
        super(CategoricalDQNModel, self).__init__(config)

    def create_tf_operations(self, config):
        super(CategoricalDQNModel, self).create_tf_operations(config)

        # Placeholders
        with tf.variable_scope('placeholder'):
            self.next_state = dict()
            for name, state in config.states.items():
                self.next_state[name] = tf.placeholder(dtype=util.tf_dtype(state.type), shape=(None,) + tuple(state.shape), name=name)

        # setup constants delta_z and z. z represents the discretized scaling over vmin -> vmax
        scaling_increment = (self.distribution_max - self.distribution_min) / (self.num_atoms - 1)  # delta_z in the paper
        quantized_steps = self.distribution_min + np.arange(self.num_atoms) * scaling_increment  # z in the paper

        num_actions = {name: action.num_actions for name, action in config.actions}

        # creating networks
        network_builder = util.get_function(fct=config.network)

        # Training network
        with tf.variable_scope('training'):
            self.training_network = NeuralNetwork(network_builder=network_builder, inputs=self.state)
            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)
            training_output = dict()
            training_output_logits = dict()

            for action in self.action:
                # for each action create an output of length num_atoms
                # this results in an array of output shape (batch_size, num_actions, num_atoms)
                # tensors are immutable so we must use lists then stack later
                actions_and_logits = []
                actions_and_probabilities = []
                for action_ind in range(num_actions[action]):
                    logits_output = layers['linear'](x=self.training_network.output, size=self.num_atoms)
                    # logits are stored for use in loss function
                    actions_and_logits.append(logits_output)
                    # softmax
                    actions_and_probabilities.append(layers['nonlinearity'](x=logits_output, name='softmax'))

                # actions_and_x shape (batch_size, num_actions, num_atoms)
                actions_and_logits = tf.stack(actions_and_logits, axis=1)
                actions_and_probabilities = tf.stack(actions_and_probabilities, axis=1)
                training_output_logits[action] = actions_and_logits
                # Q value of state is action atom probabilities * quantization steps.
                # in the paper: sum_i(z_i * p_i(x, a)) <- the a here represents all actions
                training_output[action] = tf.reduce_sum(actions_and_probabilities * quantized_steps, axis=-1)
                # training_output[action] shape = (batchsize, num_actions)
                self.action_taken[action] = tf.argmax(training_output[action], axis=1)

        # Target network
        with tf.variable_scope('target'):
            self.target_network = NeuralNetwork(network_builder=network_builder, inputs=self.next_state)
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)
            target_value = dict()
            target_output_probabilities = dict()
            target_action = dict()

            for action in self.action:
                target_actions_and_probabilities = []

                for action_ind in range(num_actions[action]):
                    target_logits_output = layers['linear'](x=self.target_network.output, size=self.num_atoms)
                    target_actions_and_probabilities.append(layers['nonlinearity'](x=target_logits_output, name='softmax'))

                target_actions_and_probabilities = tf.stack(target_actions_and_probabilities, axis=1)
                # Q value of state is action atom probabilities * quantization steps.
                # in the paper: sum_i(z_i * p_i(x, a)) <- the a here represents all actions
                target_output_probabilities[action] = target_actions_and_probabilities
                target_value[action] = tf.reduce_sum(target_actions_and_probabilities * quantized_steps, axis=-1)
                # a* in the paper, must cast argmax to int32 to use as index later
                target_action[action] = tf.cast(tf.argmax(target_value[action], axis=1), tf.int32)

        with tf.name_scope('update'):
            for action in self.action:
                # -1 because we cut off the end
                dynamic_batch_size = tf.shape(self.action[action])[0] - 1
                # project onto the supports
                # broadcast rewards and discounted quantization. Shape (batchsize, num_atoms). T_z_j in the paper
                reward = tf.expand_dims(self.reward[:-1], axis=1)
                terminal = tf.expand_dims(tf.cast(x=self.terminal[:-1], dtype=tf.float32), axis=1)
                broadcasted_rewards = reward + (1.0 - terminal) * (quantized_steps * self.discount)
                # clip into distribution_min, distribution_max
                quantized_discounted_reward = tf.clip_by_value(broadcasted_rewards, self.distribution_min, self.distribution_max)
                # compute quantization indecies. b, l, u in the paper
                closest_quantization = (quantized_discounted_reward - self.distribution_min) / scaling_increment
                lower_ind = tf.floor(closest_quantization)
                upper_ind = tf.ceil(closest_quantization)

                # vector magic here, create selections for later use
                batch_selection = tf.range(0, dynamic_batch_size)
                # tensorflow indexing is still not great, we stack these two and use gather_nd later
                target_batch_action_selection = tf.stack((batch_selection, target_action[action][1:]), axis=1)
                # tile expects a tensor of same shape, we are just repeating the selection num_atoms times across the last dimension
                batch_tiled_selection = tf.reshape(tf.tile(tf.reshape(batch_selection, (-1, 1)), [1, self.num_atoms]), [-1])
                # combine with lower and upper ind, same as zip(flatten(batch_tiled_selection), flatten(lower_ind))
                # also cast to int32 to use as index
                batch_lower_inds = tf.stack((batch_tiled_selection, tf.reshape(tf.cast(lower_ind, tf.int32), [-1])), axis=1)
                batch_upper_inds = tf.stack((batch_tiled_selection, tf.reshape(tf.cast(upper_ind, tf.int32), [-1])), axis=1)

                # distribute probability scaled by distance
                # in numpy the equivalent is target_output_probabilities[action][batch_selection, target_action]
                distance_lower = tf.gather_nd(target_output_probabilities[action], target_batch_action_selection) * (closest_quantization - lower_ind)
                distance_upper = tf.gather_nd(target_output_probabilities[action], target_batch_action_selection) * (upper_ind - closest_quantization)

                # sum distances aligned into quantized bins. m in the paper
                # scatter_nd actually sums the values into a zeros tensor instead of overwriting
                # this is pretty much a huge hack refer to https://github.com/tensorflow/tensorflow/issues/8102
                target_quantized_probabilities_lower = tf.scatter_nd(batch_lower_inds, tf.reshape(distance_lower, [-1]),
                                                                     (dynamic_batch_size, self.num_atoms))
                target_quantized_probabilities_upper = tf.scatter_nd(batch_upper_inds, tf.reshape(distance_upper, [-1]),
                                                                     (dynamic_batch_size, self.num_atoms))
                # no gradient should flow back to the target network
                target_quantized_probabilities = tf.stop_gradient(target_quantized_probabilities_lower + target_quantized_probabilities_upper)

                # now we have target probabilities loss is categorical cross entropy using logits
                # compare to the actions we actually took
                training_action_selection = tf.stack((batch_selection, self.action[action][1:]), axis=1)
                logits_for_action = tf.gather_nd(training_output_logits[action], training_action_selection)
                self.loss_per_instance = tf.nn.softmax_cross_entropy_with_logits(logits=logits_for_action, labels=target_quantized_probabilities)
                loss = tf.reduce_mean(self.loss_per_instance)
                tf.losses.add_loss(loss)

        # Update target network
        with tf.name_scope("update_target"):
            self.target_network_update = list()
            for v_source, v_target in zip(self.training_network.variables, self.target_network.variables):
                update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                self.target_network_update.append(update)

    def update_feed_dict(self, batch):
        if 'next_states' in batch:
            # if 'next_states' is given, just use given values
            feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
            feed_dict.update({next_state: batch['next_states'][name] for name, next_state in self.next_state.items()})
            feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})
            feed_dict[self.reward] = batch['rewards']
            feed_dict[self.terminal] = batch['terminals']
            feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})
        else:
            # if 'next_states' not explicitly given, assume temporally consistent sequence
            feed_dict = {state: batch['states'][name][:-1] for name, state in self.state.items()}
            feed_dict.update({next_state: batch['states'][name][1:] for name, next_state in self.next_state.items()})
            feed_dict.update({action: batch['actions'][name][:-1] for name, action in self.action.items()})
            feed_dict[self.reward] = batch['rewards'][:-1]
            feed_dict[self.terminal] = batch['terminals'][:-1]
            feed_dict.update({internal: batch['internals'][n][:-1] for n, internal in enumerate(self.internal_inputs)})
        return feed_dict

    def update_target(self):
        """
        Updates target network.
        :return:
        """
        self.session.run(self.target_network_update)
