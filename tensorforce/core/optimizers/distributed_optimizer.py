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

from tensorforce import util
from tensorforce.core.optimizers import MetaOptimizer


class DistributedOptimizer(MetaOptimizer):

    def __init__(self, optimizer):
        super(DistributedOptimizer, self).__init__(optimizer=optimizer)

    def tf_step(self, time, variables, global_variables, **kwargs):
        assert all(util.shape(global_var) == util.shape(local_var) for global_var, local_var in zip(global_variables, variables))

        local_diffs = self.optimizer.fn_step(time=time, variables=variables, **kwargs)

        with tf.control_dependencies(control_inputs=local_diffs):
            applied = self.optimizer.apply_step(variables=global_variables, diffs=local_diffs)

        with tf.control_dependencies(control_inputs=(applied,)):
            update_diffs = list()
            for global_var, local_var in zip(global_variables, variables):
                diff = global_var - local_var
                update_diffs.append(diff)

            applied = self.apply_step(variables=variables, diffs=update_diffs)

            # TODO: Update time, episode, etc (like in Synchronization)?

        with tf.control_dependencies(control_inputs=(applied,)):
            return [local_diff + update_diff for local_diff, update_diff in zip(local_diffs, update_diffs)]
