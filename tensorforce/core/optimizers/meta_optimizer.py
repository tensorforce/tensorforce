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

from tensorforce.core.optimizers import Optimizer


class MetaOptimizer(Optimizer):
    """
    A meta optimizer takes the optimization implemented by another optimizer and  
    modifies/optimizes its proposed result. For example, line search might be applied to find a  
    more optimal step size.
    """

    def __init__(self, optimizer, scope='meta-optimizer', summary_labels=(), **kwargs):
        """
        Creates a new meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
        """
        self.optimizer = Optimizer.from_spec(spec=optimizer, kwargs=kwargs)

        super(MetaOptimizer, self).__init__(scope=scope, summary_labels=summary_labels)

    def get_variables(self):
        return super(MetaOptimizer, self).get_variables() + self.optimizer.get_variables()
