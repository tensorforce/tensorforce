# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from collections import OrderedDict

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core.models import Model


class RandomModel(Model):
    """
    Utility class to return random actions of a desired shape and with given bounds.
    """

    def __init__(
        self,
        # Model
        states, actions, scope, device, saver, summarizer, execution, parallel_interactions,
        buffer_observe
    ):
        super().__init__(
            # Model
            states=states, actions=actions, scope=scope, device=device, saver=saver,
            summarizer=summarizer, execution=execution,
            parallel_interactions=parallel_interactions, buffer_observe=buffer_observe,
            exploration=None, variable_noise=None, states_preprocessing=None,
            reward_preprocessing=None
        )

    def tf_core_act(self, states, internals):
        if len(internals) > 0:
            raise TensorforceError.unexpected()

        actions = OrderedDict()
        for name, action_spec in self.actions_spec.items():
            batch_size = tf.shape(input=next(iter(states.values())))[0:1]
            shape = tf.constant(value=action_spec['shape'], dtype=tf.int32)
            shape = tf.concat(values=(batch_size, shape), axis=0)
            dtype = util.tf_dtype(dtype=action_spec['type'])

            if action_spec['type'] == 'bool':
                actions[name] = tf.math.less(
                    x=tf.random_uniform(shape=shape, dtype=util.tf_dtype(dtype='float')),
                    y=tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))
                )

            elif action_spec['type'] == 'int':
                num_values = tf.constant(value=action_spec['num_values'], dtype=dtype)
                actions[name] = tf.random_uniform(shape=shape, maxval=num_values, dtype=dtype)

            elif action_spec['type'] == 'float':
                if 'min_value' in action_spec:
                    min_value = tf.constant(value=action_spec['min_value'], dtype=dtype)
                    max_value = tf.constant(value=action_spec['max_value'], dtype=dtype)
                    actions[name] = tf.random_uniform(
                        shape=shape, minval=min_value, maxval=max_value, dtype=dtype
                    )

                else:
                    actions[name] = tf.random_uniform(shape=shape, dtype=dtype)

        return actions, OrderedDict()

    def tf_core_observe(self, states, internals, actions, terminal, reward):
        return util.no_operation()
