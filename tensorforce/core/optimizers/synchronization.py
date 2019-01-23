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
    The synchronization optimizer updates variables periodically to the value of a corresponding  
    set of source variables.
    """

    def __init__(self, name, sync_frequency=1, update_weight=1.0, summary_labels=None):
        """
        Creates a new synchronization optimizer instance.

        Args:
            sync_frequency: The interval between optimization calls actually performing a  
            synchronization step.
            update_weight: The update weight, 1.0 meaning a full assignment of the source  
            variables values.
        """
        super().__init__(name=name, summary_labels=summary_labels)

        self.sync_frequency = self.add_module(
            name='sync-frequency', module=sync_frequency, modules=parameter_modules, dtype='long'
        )

        update_weight = 1.0 if update_weight is None else update_weight
        self.update_weight = self.add_module(
            name='update-weight', module=update_weight, modules=parameter_modules, dtype='float'
        )

    def tf_initialize(self):
        super().tf_initialize()

        self.last_sync = self.add_variable(
            name='last-sync', dtype='long', shape=(), is_trainable=False,
            initializer=(-self.sync_frequency.value())
        )

    def tf_step(self, variables, source_variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            variables: List of variables to optimize.
            source_variables: List of source variables to synchronize with.
            **kwargs: Additional arguments, not used.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
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
                return [util.identity_operation(x=delta) for delta in deltas]

        def no_sync():
            deltas = list()
            for variable in variables:
                delta = tf.zeros(shape=util.shape(variable), dtype=util.tf_dtype(dtype='float'))
                deltas.append(delta)
            return deltas

        sync_frequency = self.sync_frequency.value()
        skip_sync = (timestep - self.last_sync < sync_frequency)
        return self.cond(pred=skip_sync, true_fn=no_sync, false_fn=apply_sync)
