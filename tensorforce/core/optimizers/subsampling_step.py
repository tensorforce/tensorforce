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

from tensorforce.core import parameter_modules, tf_function, tf_util
from tensorforce.core.optimizers import UpdateModifier
from tensorforce.core.utils import TensorDict


class SubsamplingStep(UpdateModifier):
    """
    Subsampling-step update modifier, which randomly samples a subset of batch instances before
    applying the given optimizer (specification key: `subsampling_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        fraction (parameter, int > 0 | 0.0 < float <= 1.0): Absolute/relative fraction of batch
            timesteps to subsample (<span style="color:#C00000"><b>required</b></span>).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, optimizer, fraction, name=None, arguments_spec=None):
        super().__init__(optimizer=optimizer, name=name, arguments_spec=arguments_spec)

        if isinstance(fraction, int):
            self.is_fraction_absolute = True
            self.fraction = self.submodule(
                name='fraction', module=fraction, modules=parameter_modules, dtype='int',
                min_value=1
            )
        else:
            self.is_fraction_absolute = False
            self.fraction = self.submodule(
                name='fraction', module=fraction, modules=parameter_modules, dtype='float',
                min_value=0.0, max_value=1.0
            )

    @tf_function(num_args=1)
    def step(self, *, arguments, **kwargs):
        if not self.is_fraction_absolute and self.fraction.is_constant(value=1.0):
            return self.optimizer.step(arguments=arguments, **kwargs)

        batch_size = tf_util.cast(x=tf.shape(input=arguments['reward'])[0], dtype='int')
        if self.is_fraction_absolute:
            fraction = self.fraction.is_constant()
            if fraction is None:
                fraction = self.fraction.value()
        else:
            fraction = self.fraction.value() * tf_util.cast(x=batch_size, dtype='float')
            fraction = tf_util.cast(x=fraction, dtype='int')
            one = tf_util.constant(value=1, dtype='int')
            fraction = tf.math.maximum(x=fraction, y=one)

        def subsampled_step():
            subsampled_arguments = TensorDict()
            indices = tf.random.uniform(
                shape=(fraction,), maxval=batch_size, dtype=tf_util.get_dtype(type='int')
            )

            if 'states' in arguments and 'horizons' in arguments:
                horizons = tf.gather(params=arguments['horizons'], indices=indices)
                starts = horizons[:, 0]
                lengths = horizons[:, 1]
                states_indices = tf.ragged.range(starts=starts, limits=(starts + lengths)).values
                function = (lambda x: tf.gather(params=x, indices=states_indices))
                subsampled_arguments['states'] = arguments['states'].fmap(function=function)
                starts = tf.math.cumsum(x=lengths, exclusive=True)
                subsampled_arguments['horizons'] = tf.stack(values=(starts, lengths), axis=1)

            for name, argument in arguments.items():
                if name not in subsampled_arguments:
                    subsampled_arguments[name] = tf.gather(params=argument, indices=indices)

            return self.optimizer.step(arguments=subsampled_arguments, **kwargs)

        def normal_step():
            return self.optimizer.step(arguments=arguments, **kwargs)

        return tf.cond(pred=(fraction < batch_size), true_fn=subsampled_step, false_fn=normal_step)
