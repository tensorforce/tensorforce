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

from tensorforce import TensorforceError, util
from tensorforce.core.models import Model


class RandomModel(Model):
    """
    Utility class to return random actions of a desired shape and with given bounds.
    """

    def __init__(
        self,
        # Model
        name, device, parallel_interactions, buffer_observe, summarizer, states, actions
    ):
        super().__init__(
            # Model
            name=name, device=None, parallel_interactions=parallel_interactions,
            buffer_observe=buffer_observe, execution=None, saver=None, summarizer=summarizer,
            states=states, internals=OrderedDict(), actions=actions, preprocessing=None,
            exploration=0.0, variable_noise=0.0, l2_regularization=0.0
        )

    def tf_core_act(self, states, internals, auxiliaries):
        if len(internals) > 0:
            raise TensorforceError.unexpected()

        actions = OrderedDict()
        for name, spec in self.actions_spec.items():
            batch_size = tf.shape(input=next(iter(states.values())))[0:1]
            shape = tf.constant(value=spec['shape'], dtype=tf.int32)
            shape = tf.concat(values=(batch_size, shape), axis=0)
            dtype = util.tf_dtype(dtype=spec['type'])

            if spec['type'] == 'bool':
                actions[name] = tf.math.less(
                    x=tf.random_uniform(shape=shape, dtype=util.tf_dtype(dtype='float')),
                    y=tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))
                )

            elif spec['type'] == 'int':
                # (Same code as for Exploration)
                int_dtype = util.tf_dtype(dtype='int')
                float_dtype = util.tf_dtype(dtype='float')

                # Action choices
                choices = list(range(spec['num_values']))
                choices_tile = ((1,) + spec['shape'] + (1,))
                choices = np.tile(A=[choices], reps=choices_tile)
                choices_shape = ((1,) + spec['shape'] + (spec['num_values'],))
                choices = tf.constant(value=choices, dtype=int_dtype, shape=choices_shape)
                ones = tf.ones(shape=(len(spec['shape']) + 1,), dtype=int_dtype)
                batch_size = tf.dtypes.cast(x=shape[0:1], dtype=int_dtype)
                multiples = tf.concat(values=(batch_size, ones), axis=0)
                choices = tf.tile(input=choices, multiples=multiples)

                # Random unmasked action
                mask = auxiliaries[name + '_mask']
                num_values = tf.math.count_nonzero(
                    input_tensor=mask, axis=-1, dtype=int_dtype
                )
                action = tf.random.uniform(shape=shape, dtype=float_dtype)
                action = tf.dtypes.cast(
                    x=(action * tf.dtypes.cast(x=num_values, dtype=float_dtype)), dtype=int_dtype
                )

                # Correct for masked actions
                choices = tf.boolean_mask(tensor=choices, mask=mask)
                offset = tf.math.cumsum(x=num_values, axis=-1, exclusive=True)
                actions[name] = tf.gather(params=choices, indices=(action + offset))

            elif spec['type'] == 'float':
                if 'min_value' in spec:
                    min_value = tf.constant(value=spec['min_value'], dtype=dtype)
                    max_value = tf.constant(value=spec['max_value'], dtype=dtype)
                    actions[name] = tf.random_uniform(
                        shape=shape, minval=min_value, maxval=max_value, dtype=dtype
                    )

                else:
                    actions[name] = tf.random_uniform(shape=shape, dtype=dtype)

        return actions, OrderedDict()

    def tf_core_observe(self, states, internals, auxiliaries, actions, terminal, reward):
        return util.no_operation()
