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
from tensorforce.core import parameter_modules
from tensorforce.core.optimizers import MetaOptimizer


class ClippedStep(MetaOptimizer):
    """
    The clipped-shep meta optimizer clips the values of the optimization step proposed by another  
    optimizer.
    """

    def __init__(self, name, optimizer, clipping_value, summary_labels=None):
        """
        Creates a new multi-step meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
            clipping_value: Clip deltas at this value.
        """
        super().__init__(name=name, optimizer=optimizer, summary_labels=summary_labels)

        self.clipping_value = self.add_module(
            name='clipping-value', module=clipping_value, modules=parameter_modules, dtype='float'
        )

    def tf_step(self, variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            variables: List of variables to optimize.
            **kwargs: Additional arguments passed on to the internal optimizer.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        deltas = self.optimizer.step(variables=variables, **kwargs)

        with tf.control_dependencies(control_inputs=deltas):
            clipping_value = self.clipping_value.value()
            clipped_deltas = list()
            exceeding_deltas = list()
            for delta in deltas:
                clipped_delta = tf.clip_by_value(
                    t=delta,
                    clip_value_min=-clipping_value,
                    clip_value_max=clipping_value
                )
                clipped_deltas.append(clipped_delta)
                exceeding_deltas.append(clipped_delta - delta)

        applied = self.apply_step(variables=variables, deltas=exceeding_deltas)

        with tf.control_dependencies(control_inputs=(applied,)):
            return [util.identity_operation(x=delta) for delta in clipped_deltas]
