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
from tensorforce.models import QModel
from tensorforce.core.networks import layers


class DQNModel(QModel):

    allows_discrete_actions = True
    allows_continuous_actions = False

    default_config = dict(
        double_dqn=False
    )

    def __init__(self, config):
        """Training logic for DQN.

        Args:
            config: 
        """
        config.default(DQNModel.default_config)
        super(DQNModel, self).__init__(config)

    def create_training_operations(self, config):
        self.training_output = dict()
        q_values = dict()
        for name, action in self.action.items():
            flat_size = util.prod(config.actions[name].shape)
            num_actions = config.actions[name].num_actions
            shape = (-1,) + config.actions[name].shape + (num_actions,)

            output = layers['linear'](x=self.training_network.output, size=(flat_size * num_actions))
            output = tf.reshape(tensor=output, shape=shape)

            self.training_output[name] = output
            self.action_taken[name] = tf.argmax(input=output, axis=-1)

            one_hot = tf.one_hot(indices=action, depth=num_actions)
            q_values[name] = tf.reduce_sum(input_tensor=(output * one_hot), axis=-1)

        return q_values

    def create_target_operations(self, config):
        target_values = dict()
        for name, action in self.action_taken.items():
            flat_size = util.prod(config.actions[name].shape)
            num_actions = config.actions[name].num_actions
            shape = (-1,) + config.actions[name].shape + (num_actions,)

            output = layers['linear'](x=self.target_network.output, size=(flat_size * num_actions))
            output = tf.reshape(tensor=output, shape=shape)

            if config.double_dqn:
                one_hot = tf.one_hot(indices=action, depth=num_actions)
                target_values[name] = tf.reduce_sum(input_tensor=(output * one_hot), axis=-1)
            else:
                target_values[name] = tf.reduce_max(input_tensor=output, axis=-1)

        return target_values
