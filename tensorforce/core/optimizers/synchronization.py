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

from tensorforce import util
from tensorforce.core import Module, parameter_modules
from tensorforce.core.optimizers import Optimizer


class Synchronization(Optimizer):
    """
    Synchronization optimizer, which updates variables periodically to the value of a corresponding  
    set of source variables (specification key: `synchronization`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        sync_frequency (parameter, int > 0): Timestep interval between updates which also perform a
            synchronization step (<span style="color:#00C000"><b>default</b></span>: every time).
        update_weight (parameter, 0.0 < float <= 1.0): Update weight
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, sync_frequency=1, update_weight=1.0, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels)

        self.sync_frequency = self.add_module(
            name='sync-frequency', module=sync_frequency, modules=parameter_modules, dtype='long'
        )

        self.update_weight = self.add_module(
            name='update-weight', module=update_weight, modules=parameter_modules, dtype='float'
        )

    def tf_initialize(self):
        super().tf_initialize()

        self.last_sync = self.add_variable(
            name='last-sync', dtype='long', shape=(), is_trainable=False, initializer=-1
        )

    def tf_step(self, variables, source_variables, **kwargs):
        assert all(
            util.shape(source) == util.shape(target)
            for source, target in zip(source_variables, variables)
        )

        timestep = Module.retrieve_tensor(name='timestep')

        def apply_sync():
            update_weight = self.update_weight.value()
            deltas = list()
            for source_variable, target_variable in zip(source_variables, variables):
                delta = update_weight * (source_variable - target_variable)
                deltas.append(delta)

            applied = self.apply_step(variables=variables, deltas=deltas)
            last_sync_updated = self.last_sync.assign(value=timestep)

            with tf.control_dependencies(control_inputs=(applied, last_sync_updated)):
                # Trivial operation to enforce control dependency
                return util.fmap(function=util.identity_operation, xs=deltas)

        def no_sync():
            deltas = list()
            for variable in variables:
                delta = tf.zeros(shape=util.shape(variable), dtype=util.tf_dtype(dtype='float'))
                deltas.append(delta)
            return deltas

        sync_frequency = self.sync_frequency.value()
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        skip_sync = tf.math.less(x=(timestep - self.last_sync), y=sync_frequency)
        skip_sync = tf.math.logical_and(
            x=skip_sync, y=tf.math.greater_equal(x=self.last_sync, y=zero)
        )

        return self.cond(pred=skip_sync, true_fn=no_sync, false_fn=apply_sync)
