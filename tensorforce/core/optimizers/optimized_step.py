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

from tensorforce import TensorForceError
from tensorforce.core.optimizers import MetaOptimizer
from tensorforce.core.optimizers.solvers import LineSearch


class OptimizedStep(MetaOptimizer):

    def __init__(self, optimizer, variables=None, max_iterations=10, accept_ratio=0.1, ls_mode='exponential', ls_parameter=0.5):
        self.solver = LineSearch(max_iterations=max_iterations, accept_ratio=accept_ratio, mode=ls_mode, parameter=ls_parameter)

        super(OptimizedStep, self).__init__(optimizer=optimizer, variables=variables)

    def tf_step(self, fn_loss, fn_reference=None, fn_compare=None, **kwargs):
        if (fn_reference is None) != (fn_compare is None):
            raise TensorForceError("Requires both arguments 'fn_reference' and 'fn_compare'!")

        if fn_reference is None:
            loss_before = fn_loss()
        else:
            reference = fn_reference()
            loss_before = fn_compare(reference=reference)

        with tf.control_dependencies(control_inputs=(loss_before,)):
            applied, diffs = self.optimizer.step(fn_loss=fn_loss, **kwargs)

        with tf.control_dependencies(control_inputs=(applied,)):
            if fn_reference is None:
                loss_step = fn_loss()
            else:
                loss_step = fn_compare(reference=reference)

        with tf.control_dependencies(control_inputs=(loss_step,)):

            def evaluate_step(x):
                with tf.control_dependencies(control_inputs=x):
                    applied = self.apply_step(diffs=x)
                with tf.control_dependencies(control_inputs=(applied,)):
                    if fn_compare is None:
                        return fn_loss()
                    else:
                        return fn_compare(reference=reference)

            return self.solver.solve(fn_x=evaluate_step, x_init=diffs, base_value=loss_before, target_value=loss_step)  # estimated_improvement=estimated_improvement)
