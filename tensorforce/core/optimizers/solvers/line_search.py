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

import tensorflow as tf

from tensorforce import util, TensorForceError
from tensorforce.core.optimizers.solvers import Iterative


class LineSearch(Iterative):
    """

    """

    def __init__(self, max_iterations, accept_ratio, mode, parameter):
        assert accept_ratio >= 0.0
        self.accept_ratio = accept_ratio

        if mode not in ('linear', 'exponential'):
            raise TensorForceError("Invalid line search mode: {}, please choose one of'linear' or 'exponential'".format(mode))
        self.mode = mode
        self.parameter = parameter

        super(LineSearch, self).__init__(max_iterations=max_iterations)

    def tf_solve(self, fn_x, x_init, base_value, target_value, estimated_improvement=None):
        return super(LineSearch, self).tf_solve(fn_x, x_init, base_value, target_value, estimated_improvement)

    def tf_initialize(self, x_init, base_value, target_value, estimated_improvement):
        if estimated_improvement is None:
            estimated_improvement = base_value
        improvement = (target_value - base_value) / tf.maximum(x=estimated_improvement, y=util.epsilon)
        last_improvement = improvement - 1.0

        if self.mode == 'linear':
            diffs = [-t * self.parameter for t in x_init]
            # self.x_incr = [t * self.parameter for t in x_init]
            self.estimated_incr = -estimated_improvement * self.parameter
            # next_x = [t + incr for t, incr in zip(x_init, self.x_incr)]
            estimated_improvement += self.estimated_incr
        elif self.mode == 'exponential':
            diffs = [-t * self.parameter for t in x_init]
            estimated_improvement *= self.parameter

        first_step = super(LineSearch, self).tf_initialize(x_init)
        return first_step + (diffs, improvement, last_improvement, base_value, estimated_improvement)

    def tf_step(self, x, iteration, diffs, improvement, last_improvement, base_value, estimated_improvement):
        x, iteration, diffs, improvement, last_improvement, base_value, estimated_improvement \
            = super(LineSearch, self).tf_step(x, iteration, diffs, improvement, last_improvement,
                                              base_value, estimated_improvement)

        x = [t + diff for t, diff in zip(x, diffs)]
        last_improvement = improvement

        if self.mode == 'linear':
            estimated_improvement += self.estimated_incr
        elif self.mode == 'exponential':
            diffs = [diff * self.parameter for diff in diffs]
            estimated_improvement *= self.parameter
        value = self.fn_x(x=diffs)
        improvement = (value - base_value) / tf.maximum(x=estimated_improvement, y=util.epsilon)

        return x, iteration, diffs, improvement, last_improvement, base_value, estimated_improvement

    def tf_next_step(self, x, iteration, diffs, improvement, last_improvement, base_value, estimated_improvement):

        def false_fn():
            value = self.fn_x(x=[-diff for diff in diffs])
            with tf.control_dependencies(control_inputs=(value,)):
                return False

        improved = tf.cond(pred=(improvement > last_improvement), true_fn=(lambda: True), false_fn=false_fn)

        next_step = super(LineSearch, self).tf_next_step(x, iteration, diffs, improvement, last_improvement, base_value, estimated_improvement)
        next_step = tf.logical_and(x=next_step, y=improved)
        next_step = tf.logical_and(x=next_step, y=(improvement < self.accept_ratio))
        return tf.logical_and(x=next_step, y=(estimated_improvement > util.epsilon))
