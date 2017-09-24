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

from tensorforce.core.optimizers import Optimizer


class MetaOptimizer(Optimizer):
    """A 'meta optimizer' receives an optimizer, obtains an optimization result,
     and then modifies the result using further heuristics. For example, the natural gradient
     optimizer obtains a result using a conjugate gradient solver, then refines the result
     using line search."""

    def __init__(self, optimizer, variables=None):
        super(MetaOptimizer, self).__init__(variables=variables)
        self.optimizer = Optimizer.from_config(config=optimizer, kwargs=dict(variables=variables))

    def minimize(self, fn_loss, **kwargs):
        if self.variables is None:
            self.variables = self.optimizer.variables = tf.trainable_variables() \
                                                        + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        return super(MetaOptimizer, self).minimize(fn_loss=fn_loss, **kwargs)
