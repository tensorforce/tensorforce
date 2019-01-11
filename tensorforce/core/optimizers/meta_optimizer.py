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

from tensorforce.core.optimizers import Optimizer


class MetaOptimizer(Optimizer):
    """
    A meta optimizer takes the optimization implemented by another optimizer and  
    modifies/optimizes its proposed result. For example, line search might be applied to find a  
    more optimal step size.
    """

    def __init__(self, name, optimizer, summary_labels=None, **kwargs):
        """
        Creates a new meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
        """
        super().__init__(name=name, summary_labels=summary_labels)

        from tensorforce.core.optimizers import optimizer_modules
        self.optimizer = self.add_module(
            name='inner-optimizer', module=optimizer, modules=optimizer_modules, **kwargs
        )
