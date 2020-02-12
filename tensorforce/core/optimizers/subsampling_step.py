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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import Module, parameter_modules
from tensorforce.core.optimizers import MetaOptimizer


class SubsamplingStep(MetaOptimizer):
    """
    Subsampling-step meta optimizer, which randomly samples a subset of batch instances before
    applying the given optimizer (specification key: `subsampling_step`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        fraction (parameter, 0.0 < float < 1.0): Fraction of batch timesteps to subsample
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, optimizer, fraction, summary_labels=None):
        super().__init__(name=name, optimizer=optimizer, summary_labels=summary_labels)

        self.fraction = self.add_module(
            name='fraction', module=fraction, modules=parameter_modules, dtype='float'
        )

    def tf_step(self, variables, arguments, **kwargs):
        # # Get some (batched) argument to determine batch size.
        # arguments_iter = iter(arguments.values())
        # some_argument = next(arguments_iter)

        # try:
        #     while not isinstance(some_argument, tf.Tensor) or util.rank(x=some_argument) == 0:
        #         if isinstance(some_argument, dict):
        #             if some_argument:
        #                 arguments_iter = iter(some_argument.values())
        #             some_argument = next(arguments_iter)
        #         elif isinstance(some_argument, list):
        #             if some_argument:
        #                 arguments_iter = iter(some_argument)
        #             some_argument = next(arguments_iter)
        #         elif some_argument is None or util.rank(x=some_argument) == 0:
        #             # Non-batched argument
        #             some_argument = next(arguments_iter)
        #         else:
        #             raise TensorforceError("Invalid argument type.")
        # except StopIteration:
        #     raise TensorforceError("Invalid argument type.")

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
        function = (lambda x: tf.gather(params=x, indices=indices))
        subsampled_arguments = util.fmap(function=function, xs=arguments)

        dependency_starts = Module.retrieve_tensor(name='dependency_starts')
        dependency_lengths = Module.retrieve_tensor(name='dependency_lengths')
        subsampled_starts = tf.gather(params=dependency_starts, indices=indices)
        subsampled_lengths = tf.gather(params=dependency_lengths, indices=indices)
        trivial_dependencies = tf.reduce_all(
            input_tensor=tf.math.equal(x=dependency_lengths, y=one), axis=0
        )

        def dependency_state_indices():
            fold = (lambda acc, args: tf.concat(
                values=(acc, tf.range(start=args[0], limit=(args[0] + args[1]))), axis=0
            ))
            return tf.foldl(
                fn=fold, elems=(subsampled_starts, subsampled_lengths), initializer=indices[:0],
                parallel_iterations=10, back_prop=False, swap_memory=False
            )

        states_indices = self.cond(
            pred=trivial_dependencies, true_fn=(lambda: indices), false_fn=dependency_state_indices
        )
        function = (lambda x: tf.gather(params=x, indices=states_indices))
        subsampled_arguments['states'] = util.fmap(function=function, xs=arguments['states'])

        subsampled_starts = tf.math.cumsum(x=subsampled_lengths, exclusive=True)
        Module.update_tensors(
            dependency_starts=subsampled_starts, dependency_lengths=subsampled_lengths
        )

        deltas = self.optimizer.step(variables=variables, arguments=subsampled_arguments, **kwargs)

        Module.update_tensors(
            dependency_starts=dependency_starts, dependency_lengths=dependency_lengths
        )

        return deltas
