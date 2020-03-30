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
from tensorforce.core import Module, parameter_modules, tf_function
from tensorforce.core.optimizers import UpdateModifier


class SubsamplingStep(UpdateModifier):
    """
    Subsampling-step update modifier, which randomly samples a subset of batch instances before
    applying the given optimizer (specification key: `subsampling_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        fraction (parameter, 0.0 <= float <= 1.0): Fraction of batch timesteps to subsample
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        optimized_module (module): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, optimizer, fraction, summary_labels=None, name=None, states_spec=None,
        internals_spec=None, auxiliaries_spec=None, actions_spec=None, optimized_module=None
    ):
        super().__init__(
            optimizer=optimizer, summary_labels=summary_labels, name=name, states_spec=states_spec,
            internals_spec=internals_spec, auxiliaries_spec=auxiliaries_spec,
            actions_spec=actions_spec, optimized_module=optimized_module
        )

        self.fraction = self.add_module(
            name='fraction', module=fraction, modules=parameter_modules, dtype='float',
            min_value=0.0, max_value=1.0
        )

    @tf_function(num_args=1)
    def step(self, arguments, **kwargs):
        arguments = OrderedDict(arguments)
        subsampled_arguments = OrderedDict()
        states = arguments.pop('states')
        horizons = arguments.pop('horizons')

        some_argument = arguments['reward']
        if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
            batch_size = tf.shape(input=some_argument, out_type=util.tf_dtype(dtype='long'))[0]
        else:
            batch_size = tf.dtypes.cast(
                x=tf.shape(input=some_argument)[0], dtype=util.tf_dtype(dtype='long')
            )
        fraction = self.fraction.value()
        num_samples = fraction * tf.dtypes.cast(x=batch_size, dtype=util.tf_dtype('float'))
        num_samples = tf.dtypes.cast(x=num_samples, dtype=util.tf_dtype('long'))
        one = tf.constant(value=1, dtype=util.tf_dtype('long'))
        num_samples = tf.maximum(x=num_samples, y=one)
        indices = tf.random.uniform(
            shape=(num_samples,), maxval=batch_size, dtype=util.tf_dtype(dtype='long')
        )

        is_one_horizons = tf.reduce_all(input_tensor=tf.math.equal(x=horizons[:, 1], y=one), axis=0)
        horizons = tf.gather(params=horizons, indices=indices)

        def subsampled_states_indices():
            fold = (lambda acc, h: tf.concat(
                values=(acc, tf.range(start=h[0], limit=(h[0] + h[1]))), axis=0
            ))
            return tf.foldl(fn=fold, elems=horizons, initializer=indices[:0])

        states_indices = self.cond(
            pred=is_one_horizons, true_fn=(lambda: indices), false_fn=subsampled_states_indices
        )
        function = (lambda x: tf.gather(params=x, indices=states_indices))
        subsampled_arguments['states'] = util.fmap(function=function, xs=states)
        subsampled_arguments['horizons'] = tf.stack(
            values=(tf.math.cumsum(x=horizons[:, 1], exclusive=True), horizons[:, 1]), axis=1
        )

        function = (lambda x: tf.gather(params=x, indices=indices))
        subsampled_arguments.update(util.fmap(function=function, xs=arguments))

        deltas = self.optimizer.step(arguments=subsampled_arguments, **kwargs)

        return deltas
