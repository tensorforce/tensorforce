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
        with tf.variable_scope('training') as training_scope:
            self.training_network = NeuralNetwork(network_builder=network_builder, inputs=self.state, summary_level=config.tf_summary_level)
            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)
            training_output_logits, training_output_probabilities, training_qval, action_taken = self._create_action_outputs(
                self.training_network.output, quantized_steps, self.num_atoms, config, self.action, num_actions
            )
            # stack to preserve action_taken shape like (batch_size, num_actions)
            for action in self.action:
                if len(action_taken[action]) > 1:
                    self.action_taken[action] = tf.stack(action_taken[action], axis=1)
                else:
                    self.action_taken[action] = action_taken[action][0]

                # summarize expected reward histogram
                if config.tf_summary_level >= 1:
                    for action_shaped in range(len(action_taken[action])):
                        for action_ind in range(num_actions[action]):
                            tf.summary.histogram('{}-{}-{}-output-distribution'.format(action, action_shaped, action_ind),
                                                 training_output_probabilities[action][action_shaped][:, action_ind] * quantized_steps)

            self.training_variables = tf.contrib.framework.get_variables(scope=training_scope)

        # Target network
        with tf.variable_scope('target') as target_scope:
            self.target_network = NeuralNetwork(network_builder=network_builder, inputs=self.next_state)
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)
            _, target_output_probabilities, target_qval, target_action = self._create_action_outputs(
                self.target_network.output, quantized_steps, self.num_atoms, config, self.action, num_actions
            )

            self.target_variables = tf.contrib.framework.get_variables(scope=target_scope)

        with tf.name_scope('update'):
            # broadcast rewards and discounted quantization. Shape (batchsize, num_atoms). T_z_j in the paper
            reward = tf.expand_dims(self.reward, axis=1)
            terminal = tf.expand_dims(tf.cast(x=self.terminal, dtype=tf.float32), axis=1)
            broadcasted_rewards = reward + (1.0 - terminal) * (quantized_steps * self.discount)
            # clip into distribution_min, distribution_max
            quantized_discounted_reward = tf.clip_by_value(broadcasted_rewards, self.distribution_min, self.distribution_max)
            # compute quantization indecies. b, l, u in the paper
            closest_quantization = (quantized_discounted_reward - self.distribution_min) / scaling_increment
            lower_ind = tf.floor(closest_quantization)
            upper_ind = tf.ceil(closest_quantization)

            # create shared selections for later use
            dynamic_batch_size = tf.shape(self.reward)[0]
            batch_selection = tf.range(0, dynamic_batch_size)
            # tile expects a tensor of same shape, we are just repeating the selection num_atoms times across the last dimension
            batch_tiled_selection = tf.reshape(tf.tile(tf.reshape(batch_selection, (-1, 1)), [1, self.num_atoms]), [-1])
            # combine with lower and upper ind, same as zip(flatten(batch_tiled_selection), flatten(lower_ind))
            # also cast to int32 to use as index
            batch_lower_inds = tf.stack((batch_tiled_selection, tf.reshape(tf.cast(lower_ind, tf.int32), [-1])), axis=1)
            batch_upper_inds = tf.stack((batch_tiled_selection, tf.reshape(tf.cast(upper_ind, tf.int32), [-1])), axis=1)

            # create loss for each action
            for action in self.action:
                # if shape of action != () we need to process each action head separately
                for action_ind in range(max([util.prod(config.actions[action].shape), 1])):
                    # project onto the supports
                    # tensorflow indexing is still not great, we stack these two and use gather_nd later
                    target_batch_action_selection = tf.stack((batch_selection, target_action[action][action_ind]), axis=1)

                    # distribute probability scaled by distance
                    # in numpy the equivalent is target_output_probabilities[action][batch_selection, target_action]
                    target_probabilities_of_action = tf.gather_nd(target_output_probabilities[action][action_ind],
                                                                  target_batch_action_selection)
                    distance_lower = target_probabilities_of_action * (closest_quantization - lower_ind)
                    distance_upper = target_probabilities_of_action * (upper_ind - closest_quantization)

                    # sum distances aligned into quantized bins. m in the paper
                    # scatter_nd actually sums the values into a zeros tensor instead of overwriting
                    # this is pretty much a huge hack refer to https://github.com/tensorflow/tensorflow/issues/8102
                    target_quantized_probabilities_lower = tf.scatter_nd(batch_lower_inds, tf.reshape(distance_lower, [-1]),
                                                                         (dynamic_batch_size, self.num_atoms))
                    target_quantized_probabilities_upper = tf.scatter_nd(batch_upper_inds, tf.reshape(distance_upper, [-1]),
                                                                         (dynamic_batch_size, self.num_atoms))
                    # no gradient should flow back to the target network
                    target_quantized_probabilities = tf.stop_gradient(target_quantized_probabilities_lower + target_quantized_probabilities_upper)

                    # we must check if input action has shape
                    if len(self.action[action].shape) > 1:
                        input_action = self.action[action][:, action_ind]
                    else:
                        input_action = self.action[action]
                    # now we have target probabilities loss is categorical cross entropy using logits
                    # compare to the actions we actually took
                    training_action_selection = tf.stack((batch_selection, input_action), axis=1)
                    logits_for_action = tf.gather_nd(training_output_logits[action][action_ind], training_action_selection)
                    self.loss_per_instance = tf.nn.softmax_cross_entropy_with_logits(logits=logits_for_action, labels=target_quantized_probabilities)
                    loss = tf.reduce_mean(self.loss_per_instance)
                    tf.losses.add_loss(loss)

                    tf.summary.scalar('cce-loss-{}-{}'.format(action, action_ind), loss)

        # Update target network
        with tf.name_scope("update_target"):
            self.target_network_update = list()
            for v_source, v_target in zip(self.training_variables, self.target_variables):
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

    @staticmethod
    def _create_action_outputs(network_output, quantized_steps, num_atoms, config, actions, num_actions):
        action_logits = dict()
        action_probabilities = dict()
        action_qvals = dict()
        action_taken = dict()
        for action in actions:
            logits = []
            probabilities = []
            qvals = []
            argmax = []
            # if shape of action != () we need to create another network head for each
            # but always create at least 1
            for shaped_action in range(max([util.prod(config.actions[action].shape), 1])):
                # for each action create an output of length num_atoms
                # this results in an array of output shape (batch_size, num_actions, num_atoms)
                # tensors are immutable so we must use lists then stack later
                actions_and_logits = []
                actions_and_probabilities = []
                for action_ind in range(num_actions[action]):
                    logits_output = layers['linear'](x=network_output, size=num_atoms)
                    # logits are stored for use in loss function
                    actions_and_logits.append(logits_output)
                    # softmax
                    actions_and_probabilities.append(layers['nonlinearity'](x=logits_output, name='softmax'))

                # actions_and_x shape (batch_size, num_actions, num_atoms)
                actions_and_logits = tf.stack(actions_and_logits, axis=1)
                actions_and_probabilities = tf.stack(actions_and_probabilities, axis=1)

                logits.append(actions_and_logits)
                probabilities.append(actions_and_probabilities)
                # Q value of state is action atom probabilities * quantization steps.
                # in the paper: sum_i(z_i * p_i(x, a)) <- the a here represents all actions
                qvals.append(tf.reduce_sum(actions_and_probabilities * quantized_steps, axis=-1))
                # qval shape = (batchsize, num_actions)
                # must cast argmax to int32 to use as index later
                argmax.append(tf.cast(tf.argmax(qvals[-1], axis=1), tf.int32))

            action_logits[action] = logits
            action_probabilities[action] = probabilities
            action_qvals[action] = qvals
            action_taken[action] = argmax
        return action_logits, action_probabilities, action_qvals, action_taken
