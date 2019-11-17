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

import numpy as np
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
        name, device, parallel_interactions, buffer_observe, seed, summarizer, config, states,
        actions,
        # ConstantModel
        action_values
    ):
        super().__init__(
            # Model
            name=name, device=None, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe, seed=seed, execution=None, saver=None,
            summarizer=summarizer, config=config, states=states, internals=OrderedDict(),
            actions=actions, preprocessing=None, exploration=0.0, variable_noise=0.0,
            l2_regularization=0.0
        )

        # check values
        self.action_values = action_values

    def tf_core_act(self, states, internals, auxiliaries):
        assert len(internals) == 0

        actions = OrderedDict()
        for name, spec in self.actions_spec.items():
            some_state = next(iter(states.values()))
            if util.tf_dtype(dtype='int') in (tf.int32, tf.int64):
                batch_size = tf.shape(input=some_state, out_type=util.tf_dtype(dtype='int'))[0:1]
            else:
                batch_size = tf.dtypes.cast(
                    x=tf.shape(input=some_state)[0:1], dtype=util.tf_dtype(dtype='int')
                )
            shape = tf.constant(value=spec['shape'], dtype=util.tf_dtype(dtype='int'))
            shape = tf.concat(values=(batch_size, shape), axis=0)
            dtype = util.tf_dtype(dtype=spec['type'])

            if spec['type'] == 'int':
                # Action choices
                int_dtype = util.tf_dtype(dtype='int')
                choices = list(range(spec['num_values']))
                choices_tile = ((1,) + spec['shape'] + (1,))
                choices = np.tile(A=[choices], reps=choices_tile)
                choices_shape = ((1,) + spec['shape'] + (spec['num_values'],))
                choices = tf.constant(value=choices, dtype=int_dtype, shape=choices_shape)
                ones = tf.ones(shape=(len(spec['shape']) + 1,), dtype=int_dtype)
                batch_size = tf.dtypes.cast(x=shape[0:1], dtype=int_dtype)
                multiples = tf.concat(values=(batch_size, ones), axis=0)
                choices = tf.tile(input=choices, multiples=multiples)

                # First unmasked action
                mask = auxiliaries[name + '_mask']
                num_values = tf.math.count_nonzero(input=mask, axis=-1, dtype=int_dtype)
                offset = tf.math.cumsum(x=num_values, axis=-1, exclusive=True)
                if self.action_values is not None and name in self.action_values:
                    action = self.action_values[name]
                    num_values = tf.math.count_nonzero(
                        input=mask[..., :action], axis=-1, dtype=int_dtype
                    )
                    action = tf.math.cumsum(x=num_values, axis=-1, exclusive=True)
                else:
                    action = tf.zeros_like(input=offset)
                choices = tf.boolean_mask(tensor=choices, mask=mask)
                actions[name] = tf.gather(params=choices, indices=(action + offset))

            elif spec['type'] == 'float' and 'min_value' in spec:
                min_value = spec['min_value']
                max_value = spec['max_value']
                if self.action_values is not None and name in self.action_values:
                    assert min_value <= self.action_values[name] <= max_value
                    action = self.action_values[name]
                else:
                    action = min_value + 0.5 * (max_value - min_value)
                actions[name] = tf.fill(dims=shape, value=tf.constant(value=action, dtype=dtype))

            elif self.action_values is not None and name in self.action_values:
                value = self.action_values[name]
                actions[name] = tf.fill(dims=shape, value=tf.constant(value=value, dtype=dtype))

            else:
                actions[name] = tf.zeros(shape=shape, dtype=dtype)

        return actions, OrderedDict()

    def tf_core_observe(self, states, internals, auxiliaries, actions, terminal, reward):
        return util.no_operation()
