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
    The clipped-step optimizer clips the values of the optimization step proposed by another  
    optimizer.
    """

    def __init__(self, name, optimizer, clipping_value, summary_labels=None):
        """
        Clipped-step optimizer constructor.

        Args:
            clipping_value (parameter, float > 0.0): Clipping value (**required**).
            mode...
        """
        super().__init__(name=name, optimizer=optimizer, summary_labels=summary_labels)

        self.clipping_value = self.add_module(
            name='clipping-value', module=clipping_value, modules=parameter_modules, dtype='float'
        )

    def tf_step(self, variables, **kwargs):
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
            return util.fmap(function=util.identity_operation, xs=clipped_deltas)
