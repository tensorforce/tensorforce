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

    def __init__(self, *, states, actions, name, device, parallel_interactions, summarizer, config):
        super().__init__(
            states=states, actions=actions, l2_regularization=0.0, name=name, device=None,
            parallel_interactions=parallel_interactions, saver=None, summarizer=summarizer,
            config=config
        )

    @tf_function(num_args=3)
    def core_act(self, *, states, internals, auxiliaries, parallel, independent):
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
                num_unmasked = tf.math.reduce_sum(input_tensor=mask, axis=(spec.rank + 1))
                masked_offset = tf.math.cumsum(x=num_unmasked, axis=spec.rank, exclusive=True)
                uniform = tf.random.uniform(shape=shape, dtype=tf_util.get_dtype(type='float'))
                num_unmasked = tf_util.cast(x=num_unmasked, dtype='float')
                random_offset = tf_util.cast(x=(uniform * num_unmasked), dtype='int')
                actions[name] = tf.gather(params=choices, indices=(masked_offset + random_offset))

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
