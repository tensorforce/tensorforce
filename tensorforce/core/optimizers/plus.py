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

import tensorforce.core
from tensorforce.core import tf_function
from tensorforce.core.optimizers import Optimizer


class Plus(Optimizer):
    """
    Additive combination of two optimizers (specification key: `plus`).

    Args:
        optimizer1 (specification): First optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        optimizer2 (specification): Second optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, optimizer1, optimizer2, name=None, arguments_spec=None):
        super().__init__(name=name, arguments_spec=arguments_spec)

        self.optimizer1 = self.submodule(
            name=(name + '1'), module=optimizer1, modules=tensorforce.core.optimizer_modules,
            arguments_spec=self.arguments_spec
        )
        self.optimizer2 = self.submodule(
            name=(name + '2'), module=optimizer2, modules=tensorforce.core.optimizer_modules,
            arguments_spec=self.arguments_spec
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, **kwargs):
        deltas1 = self.optimizer1.step(arguments=arguments, **kwargs)

        with tf.control_dependencies(control_inputs=deltas1):
            deltas2 = self.optimizer2.step(arguments=arguments, **kwargs)

        with tf.control_dependencies(control_inputs=deltas2):
            return [delta1 + delta2 for delta1, delta2 in zip(deltas1, deltas2)]
