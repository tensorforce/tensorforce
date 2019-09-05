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

import tensorforce.core
from tensorforce.core.optimizers import Optimizer


class Plus(Optimizer):
    """
    Additive combination of two optimizers (specification key: `plus`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer1 (specification): First optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        optimizer2 (specification): Second optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, optimizer1, optimizer2, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels)

        self.optimizer1 = self.add_module(
            name='first-optimizer', module=optimizer1, modules=tensorforce.core.optimizer_modules
        )
        self.optimizer2 = self.add_module(
            name='second-optimizer', module=optimizer2, modules=tensorforce.core.optimizer_modules
        )

    def tf_step(self, **kwargs):
        deltas1 = self.optimizer1.step(**kwargs)

        with tf.control_dependencies(control_inputs=deltas1):
            deltas2 = self.optimizer2.step(**kwargs)

        with tf.control_dependencies(control_inputs=deltas2):
            return [delta1 + delta2 for delta1, delta2 in zip(deltas1, deltas2)]
