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
    """
    The optimized-step meta optimizer applies line search to the proposed optimization step of  
    another optimizer to find a more optimal step size.
    """

    def __init__(
        self,
        optimizer,
        ls_max_iterations=10,
        ls_accept_ratio=0.9,
        ls_mode='exponential',
        ls_parameter=0.5,
        ls_unroll_loop=False
    ):
        """
        Creates a new optimized step meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
            ls_max_iterations: Maximum number of line search iterations.
            ls_accept_ratio: Line search acceptance ratio.
            ls_mode: Line search mode, see LineSearch solver.
            ls_parameter: Line search parameter, see LineSearch solver.
            ls_unroll_loop: Unroll line search loop if true.
        """
        super(OptimizedStep, self).__init__(optimizer=optimizer)

        self.solver = LineSearch(
            max_iterations=ls_max_iterations,
            accept_ratio=ls_accept_ratio,
            mode=ls_mode,
            parameter=ls_parameter,
            unroll_loop=ls_unroll_loop
        )

    def tf_step(self, time, variables, fn_loss, fn_reference=None, fn_compare=None, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            fn_loss: A callable returning the loss of the current model.
            fn_reference: A callable returning the reference values necessary for comparison.
            fn_compare: A callable comparing the current model to the reference model given by  
                its values.
            **kwargs: Additional arguments passed on to the internal optimizer.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        if (fn_reference is None) != (fn_compare is None):
            raise TensorForceError("Requires both arguments 'fn_reference' and 'fn_compare'!")

        if fn_reference is None:
            # Negative value since line search maximizes.
            loss_before = -fn_loss()
        else:
            reference = fn_reference()
            loss_before = fn_compare(reference=reference)

        with tf.control_dependencies(control_inputs=(loss_before,)):
            deltas = self.optimizer.step(
                time=time,
                variables=variables,
                fn_loss=fn_loss,
                fn_reference=fn_reference,
                fn_compare=fn_compare,
                return_estimated_improvement=True,
                **kwargs
            )

            if isinstance(deltas, tuple):
                # If 'return_estimated_improvement' argument exists.
                if len(deltas) != 2:
                    raise TensorForceError("Unexpected output of internal optimizer.")
                deltas, estimated_improvement = deltas
                # Negative value since line search maximizes.
                estimated_improvement = -estimated_improvement
            else:
                estimated_improvement = None

        with tf.control_dependencies(control_inputs=deltas):
            if fn_reference is None:
                # Negative value since line search maximizes.
                loss_step = -fn_loss()
            else:
                loss_step = fn_compare(reference=reference)

        with tf.control_dependencies(control_inputs=(loss_step,)):

            def evaluate_step(x):
                with tf.control_dependencies(control_inputs=x):
                    applied = self.apply_step(variables=variables, deltas=x)
                with tf.control_dependencies(control_inputs=(applied,)):
                    if fn_compare is None:
                        # Negative value since line search maximizes.
                        return -fn_loss()
                    else:
                        return fn_compare(reference=reference)

            return self.solver.solve(
                fn_x=evaluate_step,
                x_init=deltas,
                base_value=loss_before,
                target_value=loss_step,
                estimated_improvement=estimated_improvement
            )
