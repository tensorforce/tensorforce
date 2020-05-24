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
        fraction (parameter, 0.0 <= float <= 1.0): Fraction of batch timesteps to subsample
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, optimizer, fraction, name=None, arguments_spec=None):
        super().__init__(optimizer=optimizer, name=name, arguments_spec=arguments_spec)

        self.fraction = self.submodule(
            name='fraction', module=fraction, modules=parameter_modules, dtype='float',
            min_value=0.0, max_value=1.0
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, **kwargs):
        if 'states' in arguments and 'horizons' in arguments:
            states = arguments['states']
            horizons = arguments['horizons']
        else:
            states = None

        # TODO: item, but not states/horizons
        batch_size = tf_util.cast(x=tf.shape(input=arguments['reward'])[0], dtype='int')
        fraction = self.fraction.value()
        num_samples = fraction * tf_util.cast(x=batch_size, dtype='float')
        num_samples = tf_util.cast(x=num_samples, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        num_samples = tf.math.maximum(x=num_samples, y=one)
        indices = tf.random.uniform(
            shape=(num_samples,), maxval=batch_size, dtype=tf_util.get_dtype(type='int')
        )

        subsampled_arguments = TensorDict()

        if states is not None:
            horizons = tf.gather(params=horizons, indices=indices)
            starts = horizons[:, 0]
            lengths = horizons[:, 1]
            states_indices = tf.ragged.range(starts=starts, limits=(starts + lengths)).values
            function = (lambda x: tf.gather(params=x, indices=states_indices))
            subsampled_arguments['states'] = states.fmap(function=function)
            starts = tf.math.cumsum(x=lengths, exclusive=True)
            subsampled_arguments['horizons'] = tf.stack(values=(starts, lengths), axis=1)

        for name, argument in arguments.items():
            if states is None or (not name.startswith('states') and name != 'horizons'):
                subsampled_arguments[name] = tf.gather(params=argument, indices=indices)

        deltas = self.optimizer.step(arguments=subsampled_arguments, **kwargs)

        return deltas
