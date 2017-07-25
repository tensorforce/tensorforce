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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf

from tensorforce import util
from tensorforce.models import Model
from tensorforce.core.networks import NeuralNetwork, layers


class NAFModel(Model):

    allows_discrete_actions = False
    allows_continuous_actions = True

    default_config = dict(
        update_target_weight=1.0,
        clip_gradients=0.0
    )

    def __init__(self, config):
        """
        Training logic for NAFs.

        :param config: Configuration parameters
        """
        config.default(NAFModel.default_config)
        super(NAFModel, self).__init__(config)

    def create_tf_operations(self, config):
        super(NAFModel, self).create_tf_operations(config)
        num_actions = sum(util.prod(config.actions[name].shape) for name in sorted(self.action))

        # Get hidden layers from network generator, then add NAF outputs, same for target network
        with tf.variable_scope('training'):
            network_builder = util.get_function(fct=config.network)
            self.training_network = NeuralNetwork(network_builder=network_builder, inputs=self.state)
            self.internal_inputs.extend(self.training_network.internal_inputs)
            self.internal_outputs.extend(self.training_network.internal_outputs)
            self.internal_inits.extend(self.training_network.internal_inits)

        with tf.variable_scope('training_outputs') as scope:
            # Action outputs
            flat_mean = layers['linear'](x=self.training_network.output, size=num_actions)
            n = 0
            for name in sorted(self.action):
                shape = config.actions[name].shape
                self.action_taken[name] = tf.reshape(tensor=flat_mean[:, n: n + util.prod(shape)], shape=((-1,) + shape))
                n += util.prod(shape)

            # Advantage computation
            # Network outputs entries of lower triangular matrix L
            lower_triangular_size = num_actions * (num_actions + 1) // 2
            l_entries = layers['linear'](x=self.training_network.output, size=lower_triangular_size)

            l_matrix = tf.exp(x=tf.map_fn(fn=tf.diag, elems=l_entries[:, :num_actions]))

            if num_actions > 1:
                offset = num_actions
                l_columns = list()
                for zeros, size in enumerate(xrange(num_actions - 1, -1, -1), 1):
                    column = tf.pad(tensor=l_entries[:, offset: offset + size], paddings=((0, 0), (zeros, 0)))
                    l_columns.append(column)
                    offset += size
                l_matrix += tf.stack(values=l_columns, axis=1)

            # P = LL^T
            p_matrix = tf.matmul(a=l_matrix, b=tf.transpose(a=l_matrix, perm=(0, 2, 1)))

            flat_action = list()
            for name in sorted(self.action):
                shape = config.actions[name].shape
                flat_action.append(tf.reshape(tensor=self.action[name], shape=(-1, util.prod(shape))))
            flat_action = tf.concat(values=flat_action, axis=1)
            difference = flat_action - flat_mean

            # A = -0.5 (a - mean)P(a - mean)
            advantage = tf.matmul(a=p_matrix, b=tf.expand_dims(input=difference, axis=2))
            advantage = tf.matmul(a=tf.expand_dims(input=difference, axis=1), b=advantage)
            advantage = tf.squeeze(input=(-advantage / 2.0), axis=2)

            # Q = A + V
            # State-value function
            value = layers['linear'](x=self.training_network.output, size=num_actions)
            q_value = value + advantage
            training_output_vars = tf.contrib.framework.get_variables(scope=scope)

        with tf.variable_scope('target'):
            network_builder = util.get_function(fct=config.network)
            self.target_network = NeuralNetwork(network_builder=network_builder, inputs=self.state)
            self.internal_inputs.extend(self.target_network.internal_inputs)
            self.internal_outputs.extend(self.target_network.internal_outputs)
            self.internal_inits.extend(self.target_network.internal_inits)

        with tf.variable_scope('target_outputs') as scope:
            # State-value function
            target_value = layers['linear'](x=self.target_network.output, size=num_actions)
            target_output_vars = tf.contrib.framework.get_variables(scope=scope)

        with tf.name_scope('update'):
            reward = tf.expand_dims(input=self.reward[:-1], axis=1)
            terminal = tf.expand_dims(input=tf.cast(x=self.terminal[:-1], dtype=tf.float32), axis=1)
            q_target = reward + (1.0 - terminal) * config.discount * target_value[1:]
            delta = q_target - q_value[:-1]
            delta = tf.reduce_mean(input_tensor=delta, axis=1)
            self.loss_per_instance = tf.square(x=delta)

            # We observe issues with numerical stability in some tests, gradient clipping can help
            if config.clip_gradients > 0.0:
                huber_loss = tf.where(condition=(tf.abs(delta) < config.clip_gradients), x=(0.5 * self.loss_per_instance), y=(tf.abs(delta) - 0.5))
                loss = tf.reduce_mean(input_tensor=huber_loss, axis=0)
            else:
                loss = tf.reduce_mean(input_tensor=self.loss_per_instance, axis=0)
            tf.losses.add_loss(loss)

        with tf.name_scope('update_target'):
            # Combine hidden layer variables and output layer variables
            training_vars = self.training_network.variables + training_output_vars
            target_vars = self.target_network.variables + target_output_vars

            self.target_network_update = list()
            for v_source, v_target in zip(training_vars, target_vars):
                update = v_target.assign_sub(config.update_target_weight * (v_target - v_source))
                self.target_network_update.append(update)

    def update_target(self):
        """
        Updates target network.

        :return:
        """
        self.session.run(self.target_network_update)
