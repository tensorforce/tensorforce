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

from tensorforce.core import parameter_modules, TensorSpec, tf_function, tf_util
from tensorforce.core.optimizers import Optimizer


class Synchronization(Optimizer):
    """
    Synchronization optimizer, which updates variables periodically to the value of a corresponding
    set of source variables (specification key: `synchronization`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        update_weight (parameter, 0.0 < float <= 1.0): Update weight
            (<span style="color:#C00000"><b>required</b></span>).
        sync_frequency (parameter, int >= 1): Interval between updates which also perform a
            synchronization step (<span style="color:#00C000"><b>default</b></span>: every update).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, update_weight, sync_frequency=None, name=None, arguments_spec=None):
        super().__init__(name=name, arguments_spec=arguments_spec)

        self.update_weight = self.submodule(
            name='update_weight', module=update_weight, modules=parameter_modules, dtype='float',
            min_value=0.0, max_value=1.0
        )

        if sync_frequency is None:
            sync_frequency = 1
        self.sync_frequency = self.submodule(
            name='sync_frequency', module=sync_frequency, modules=parameter_modules,
            dtype='int', min_value=1
        )

    def initialize(self):
        super().initialize()

        if not self.sync_frequency.is_constant(value=1):
            self.next_sync = self.variable(
                name='next-sync', spec=TensorSpec(type='int'), initializer='zeros',
                is_trainable=False, is_saved=True
            )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        assert 'source_variables' in kwargs
        source_variables = kwargs['source_variables']

        assert all(
            tf_util.shape(x=source) == tf_util.shape(x=target)
            for source, target in zip(source_variables, variables)
        )

        one = tf_util.constant(value=1, dtype='int')

        def apply_sync():
            dependencies = list()
            if not self.sync_frequency.is_constant(value=1):
                dependencies.append(self.next_sync.assign(
                    value=self.sync_frequency.value(), read_value=False
                ))

            with tf.control_dependencies(control_inputs=dependencies):
                deltas = list()
                assignments = list()
                if self.update_weight.is_constant(value=1.0):
                    for source_var, target_var in zip(source_variables, variables):
                        deltas.append(source_var - target_var)
                        assignments.append(target_var.assign(value=source_var, read_value=False))
                else:
                    update_weight = self.update_weight.value()
                    for source_var, target_var in zip(source_variables, variables):
                        delta = update_weight * (source_var - target_var)
                        deltas.append(delta)
                        assignments.append(target_var.assign_add(delta=delta, read_value=False))

            with tf.control_dependencies(control_inputs=assignments):
                # Trivial operation to enforce control dependency
                return [tf_util.identity(input=delta) for delta in deltas]

        def no_sync():
            next_sync_updated = self.next_sync.assign_sub(delta=one, read_value=False)

            with tf.control_dependencies(control_inputs=(next_sync_updated,)):
                deltas = list()
                for variable in variables:
                    delta = tf_util.zeros(shape=tf_util.shape(x=variable), dtype='float')
                    deltas.append(delta)
                return deltas

        if self.sync_frequency.is_constant(value=1):
            return apply_sync()

        else:
            skip_sync = tf.math.greater(x=self.next_sync, y=one)
            return tf.cond(pred=skip_sync, true_fn=no_sync, false_fn=apply_sync)
