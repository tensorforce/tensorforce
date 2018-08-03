# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce import util
from tensorforce.core.optimizers import Optimizer


class Synchronization(Optimizer):
    """
    The synchronization optimizer updates variables periodically to the value of a corresponding  
    set of source variables.
    """

    def __init__(self, sync_frequency=1, update_weight=1.0, scope='synchronization', summary_labels=()):
        """
        Creates a new synchronization optimizer instance.

        Args:
            sync_frequency: The interval between optimization calls actually performing a  
            synchronization step.
            update_weight: The update weight, 1.0 meaning a full assignment of the source  
            variables values.
        """
        assert isinstance(sync_frequency, int) and sync_frequency > 0
        self.sync_frequency = sync_frequency

        assert isinstance(update_weight, float) and update_weight > 0.0
        self.update_weight = update_weight

        super(Synchronization, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_step(self, time, variables, source_variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            source_variables: List of source variables to synchronize with.
            **kwargs: Additional arguments, not used.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        assert all(util.shape(source) == util.shape(target) for source, target in zip(source_variables, variables))

        last_sync = tf.get_variable(
            name='last-sync',
            shape=(),
            dtype=tf.int64,
            initializer=tf.constant_initializer(value=(-self.sync_frequency), dtype=tf.int64),
            trainable=False
        )

        def sync():
            deltas = list()
            for source_variable, target_variable in zip(source_variables, variables):
                delta = self.update_weight * (source_variable - target_variable)
                deltas.append(delta)

            applied = self.apply_step(variables=variables, deltas=deltas)
            last_sync_updated = last_sync.assign(value=time)

            with tf.control_dependencies(control_inputs=(applied, last_sync_updated)):
                # Trivial operation to enforce control dependency
                return [delta + 0.0 for delta in deltas]

        def no_sync():
            deltas = list()
            for variable in variables:
                delta = tf.zeros(shape=util.shape(variable))
                deltas.append(delta)
            return deltas

        do_sync = (time - last_sync >= self.sync_frequency)
        return tf.cond(pred=do_sync, true_fn=sync, false_fn=no_sync)
