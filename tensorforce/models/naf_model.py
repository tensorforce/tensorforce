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
from tensorforce.models import QModel
from tensorforce.core.networks import layers


class NAFModel(QModel):

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

    def create_training_operations(self, config):
        num_actions = sum(util.prod(config.actions[name].shape) for name in sorted(self.action))

        # Get hidden layers from network generator, then add NAF outputs, same for target network
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

        q_values = dict()
        n = 0
        for name in sorted(self.action):
            shape = (-1,) + config.actions[name].shape
            flat_size = util.prod(shape[1:])
            q_values[name] = tf.reshape(tensor=q_value[:, n: n + flat_size], shape=shape)
            n += flat_size
        return q_values

    def create_target_operations(self, config):
        # State-value function
        num_actions = sum(util.prod(config.actions[name].shape) for name in sorted(self.action))
        target_value = layers['linear'](x=self.target_network.output, size=num_actions)

        target_values = dict()
        n = 0
        for name in sorted(self.action):
            shape = (-1,) + config.actions[name].shape
            flat_size = util.prod(shape[1:])
            target_values[name] = tf.reshape(tensor=target_value[:, n: n + flat_size], shape=shape)
            n += flat_size
        return target_values
