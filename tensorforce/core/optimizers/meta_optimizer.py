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
    A 'meta optimizer' receives an optimizer, obtains an optimization result,
    and then modifies the result using further heuristics. For example, the natural gradient
    optimizer obtains a result using a conjugate gradient solver, then refines the result
    using line search.
    """

    def __init__(self, optimizer):
        super(MetaOptimizer, self).__init__()

        self.optimizer = Optimizer.from_spec(spec=optimizer)

    def minimize(self, time, variables, **kwargs):
        return super(MetaOptimizer, self).minimize(time=time, variables=variables, **kwargs)
