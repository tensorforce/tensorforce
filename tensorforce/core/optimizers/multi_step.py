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

from six.moves import xrange
import tensorflow as tf

from tensorforce.core.optimizers import MetaOptimizer


class MultiStep(MetaOptimizer):

    def __init__(self, optimizer, num_steps=5):
        super(MultiStep, self).__init__(optimizer=optimizer)

        assert isinstance(num_steps, int) and num_steps > 0
        self.num_steps = num_steps

    def tf_step(self, time, variables, fn_loss, **kwargs):
        overall_diffs = None
        diffs = ()
        for _ in xrange(self.num_steps):

            with tf.control_dependencies(control_inputs=diffs):
                diffs = self.optimizer.fn_step(time=time, variables=variables, fn_loss=fn_loss, **kwargs)

                if overall_diffs is None:
                    overall_diffs = diffs
                else:
                    overall_diffs = [diff1 + diff2 for diff1, diff2 in zip(overall_diffs, diffs)]

        return overall_diffs
