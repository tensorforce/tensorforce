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

import tensorforce.core
from tensorforce.core.optimizers import Optimizer


class MetaOptimizer(Optimizer):
    """
    Meta optimizer, which takes the update mechanism implemented by another optimizer and modifies
    it.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, optimizer, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels)

        self.optimizer = self.add_module(
            name='inner-optimizer', module=optimizer, modules=tensorforce.core.optimizer_modules
        )
