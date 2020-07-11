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

import tensorforce.core
from tensorforce.core.optimizers import Optimizer


class UpdateModifier(Optimizer):
    """
    Update modifier, which takes the update mechanism implemented by another optimizer and modifies
    it.

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, optimizer, name=None, arguments_spec=None):
        super().__init__(name=name, arguments_spec=arguments_spec)

        self.optimizer = self.submodule(
            name=name, module=optimizer, modules=tensorforce.core.optimizer_modules,
            arguments_spec=self.arguments_spec
        )
