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

from tensorforce import util
from tensorforce.core.models import Model


class ConstantModel(Model):
    """
    Utility class to return constant actions of a desired shape and with given bounds.
    """

    def __init__(
        self,
        # Model
        states, actions, scope, device, saver, summarizer, execution, parallel_interactions,
        buffer_observe,
        # ConstantModel
        action_values
    ):
        super().__init__(
            # Model
            states=states, actions=actions, scope=scope, device=device, saver=saver,
            summarizer=summarizer, execution=execution,
            parallel_interactions=parallel_interactions, buffer_observe=buffer_observe,
            exploration=None, variable_noise=None, states_preprocessing=None,
            reward_preprocessing=None
        )

        # check values
        self.action_values = action_values

    def tf_core_act(self, states, internals):
        assert len(internals) == 0

        actions = OrderedDict()
        for name, action_spec in self.actions_spec.items():
            batch_size = tf.shape(
                input=next(iter(states.values())), out_type=util.tf_dtype(dtype='int')
            )[0:1]
            shape = tf.constant(value=action_spec['shape'], dtype=util.tf_dtype(dtype='int'))
            shape = tf.concat(values=(batch_size, shape), axis=0)
            dtype = util.tf_dtype(dtype=action_spec['type'])

            if self.action_values is not None and name in self.action_values:
                value = self.action_values[name]
                actions[name] = tf.fill(dims=shape, value=tf.constant(value=value, dtype=dtype))

            elif action_spec['type'] == 'float' and 'min_value' in action_spec:
                min_value = action_spec['min_value']
                max_value = action_spec['max_value']
                mean = min_value + 0.5 * (max_value - min_value)
                actions[name] = tf.fill(dims=shape, value=tf.constant(value=mean, dtype=dtype))

            else:
                actions[name] = tf.zeros(shape=shape, dtype=dtype)

        return actions, OrderedDict()

    def tf_core_observe(self, states, internals, actions, terminal, reward):
        return util.no_operation()
