# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import tensorflow as tf

from tensorforce.core import TensorDict, tf_function, tf_util
from tensorforce.core.models import Model


class RandomModel(Model):
    """
    Utility class to return random actions of a desired shape and with given bounds.
    """

    def __init__(self, *, states, actions, parallel_interactions, summarizer, config):
        super().__init__(
            states=states, actions=actions, l2_regularization=0.0,
            parallel_interactions=parallel_interactions, saver=None, summarizer=summarizer,
            config=config
        )

    @tf_function(num_args=5)
    def core_act(self, *, states, internals, auxiliaries, parallel, deterministic, independent):
        assert len(internals) == 0

        actions = TensorDict()
        for name, spec in self.actions_spec.items():
            shape = tf.concat(values=(
                tf_util.cast(x=tf.shape(input=states.value())[:1], dtype='int'),
                tf_util.constant(value=spec.shape, dtype='int')
            ), axis=0)

            if spec.type == 'bool':
                # Random bool action: uniform[True, False]
                half = tf_util.constant(value=0.5, dtype='float')
                uniform = tf.random.uniform(shape=shape, dtype=tf_util.get_dtype(type='float'))
                actions[name] = (uniform < half)

            elif self.config.enable_int_action_masking and spec.type == 'int' and \
                    spec.num_values is not None:
                # Random masked action: uniform[unmasked]
                # (Similar code as for Model.apply_exploration)
                mask = auxiliaries[name]['mask']
                choices = tf_util.constant(
                    value=list(range(spec.num_values)), dtype=spec.type,
                    shape=(tuple(1 for _ in spec.shape) + (1, spec.num_values))
                )
                one = tf_util.constant(value=1, dtype='int', shape=(1,))
                multiples = tf.concat(values=(shape, one), axis=0)
                choices = tf.tile(input=choices, multiples=multiples)
                choices = tf.boolean_mask(tensor=choices, mask=mask)
                mask = tf_util.cast(x=mask, dtype='int')
                num_valid = tf.math.reduce_sum(input_tensor=mask, axis=(spec.rank + 1))
                num_valid = tf.reshape(tensor=num_valid, shape=(-1,))
                masked_offset = tf.math.cumsum(x=num_valid, axis=0, exclusive=True)
                uniform = tf.random.uniform(shape=shape, dtype=tf_util.get_dtype(type='float'))
                uniform = tf.reshape(tensor=uniform, shape=(-1,))
                num_valid = tf_util.cast(x=num_valid, dtype='float')
                random_offset = tf.dtypes.cast(x=(uniform * num_valid), dtype=tf.dtypes.int64)
                action = tf.gather(params=choices, indices=(masked_offset + random_offset))
                actions[name] = tf.reshape(tensor=action, shape=shape)

            elif spec.type != 'bool' and spec.min_value is not None:
                if spec.max_value is not None:
                    # Random bounded action: uniform[min_value, max_value]
                    actions[name] = tf.random.uniform(
                        shape=shape, minval=spec.min_value, maxval=spec.max_value,
                        dtype=spec.tf_type()
                    )

                else:
                    # Random left-bounded action: not implemented
                    raise NotImplementedError

            elif spec.type != 'bool' and spec.max_value is not None:
                # Random right-bounded action: not implemented
                raise NotImplementedError

            else:
                # Random unbounded int/float action
                actions[name] = tf.random.normal(shape=shape, dtype=spec.tf_type())

        return actions, TensorDict()
