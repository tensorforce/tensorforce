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

from tensorforce.models import Model
import tensorflow as tf


class RandomModel(Model):
    """
    Utility class to return (uniform) random actions of a desired shape
    and with given bounds.
    """

    def tf_actions_and_internals(self, states, internals, deterministic):
        actions = dict()

        for name, action in self.actions_spec.items():
            if action['type'] == 'bool':
                action = tf.random_uniform(shape=action.shape)
                actions[name] = tf.less(x=action, y=0.5)
            elif action['type'] == 'int':
                action = tf.random_uniform(shape=action.shape) * action['num_actions']
                actions[name] = tf.cast(tf.floor(action), tf.int32)
            elif action['type'] == 'float':
                if 'min_value' in action:
                    actions[name] = tf.random_uniform(
                        shape=action.shape,
                        minval=action.min_value,
                        maxval=action.max_value,
                    )
                else:
                    actions[name] = tf.random_normal(shape=action.shape, name='random-action')

        return actions, internals

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward):
        # Nothing to be done here, loss is 0.
        return tf.zeros_like(tensor=reward)

    def __init__(self, states_spec, actions_spec, config):
        super(RandomModel).__init__(states_spec, actions_spec, config)
