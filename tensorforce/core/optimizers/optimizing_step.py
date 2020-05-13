# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce import TensorforceError
from tensorforce.core import TensorDict, TensorSpec, TensorsSpec, tf_function, tf_util
from tensorforce.core.optimizers import UpdateModifier
from tensorforce.core.optimizers.solvers import solver_modules


class OptimizingStep(UpdateModifier):
    """
    Optimizing-step update modifier, which applies line search to the given optimizer to find a more
    optimal step size (specification key: `optimizing_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        ls_max_iterations (parameter, int >= 0): Maximum number of line search iterations
            (<span style="color:#00C000"><b>default</b></span>: 10).
        ls_accept_ratio (parameter, 0.0 <= float <= 1.0): Line search acceptance ratio
            (<span style="color:#00C000"><b>default</b></span>: 0.9).
        ls_mode ('exponential' | 'linear'): Line search mode, see line search solver
            (<span style="color:#00C000"><b>default</b></span>: 'exponential').
        ls_parameter (parameter, 0.0 <= float <= 1.0): Line search parameter, see line search solver
            (<span style="color:#00C000"><b>default</b></span>: 0.5).
        ls_unroll_loop (bool): Whether to unroll the line search loop
            (<span style="color:#00C000"><b>default</b></span>: false).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, optimizer, ls_max_iterations=10, ls_accept_ratio=0.9, ls_mode='exponential',
        ls_parameter=0.5, ls_unroll_loop=False, summary_labels=None, name=None, arguments_spec=None
    ):
        super().__init__(
            optimizer=optimizer, summary_labels=summary_labels, name=name,
            arguments_spec=arguments_spec
        )

        self.line_search = self.add_module(
            name='line_search', module='line_search', modules=solver_modules,
            max_iterations=ls_max_iterations, accept_ratio=ls_accept_ratio, mode=ls_mode,
            parameter=ls_parameter, unroll_loop=ls_unroll_loop
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        # Negative value since line search maximizes
        loss_before = -fn_loss(**arguments.to_kwargs())

        with tf.control_dependencies(control_inputs=(loss_before,)):
            deltas = self.optimizer.step(
                arguments=arguments, variables=variables, fn_loss=fn_loss, **kwargs,
                return_estimated_improvement=True
            )

            if isinstance(deltas, tuple):
                # If 'return_estimated_improvement' argument exists.
                if len(deltas) != 2:
                    raise TensorforceError("Unexpected output of internal optimizer.")
                deltas, estimated_improvement = deltas
                # Negative value since line search maximizes.
                estimated_improvement = -estimated_improvement
            else:
                # TODO: Is this a good alternative?
                estimated_improvement = tf.math.abs(x=loss_before)

        with tf.control_dependencies(control_inputs=deltas):
            # Negative value since line search maximizes.
            loss_step = -fn_loss(**arguments.to_kwargs())

        with tf.control_dependencies(control_inputs=(loss_step,)):

            def evaluate_step(deltas):
                with tf.control_dependencies(control_inputs=list(deltas.values())):
                    assignments = list()
                    for variable, delta in zip(variables, deltas.values()):
                        assignments.append(variable.assign_add(delta=delta, read_value=False))
                with tf.control_dependencies(control_inputs=assignments):
                    # Negative value since line search maximizes.
                    return -fn_loss(**arguments.to_kwargs())

            values_spec = TensorsSpec((
                (var.name, TensorSpec(type=tf_util.dtype(x=var), shape=tf_util.shape(x=var)))
                for var in variables
            ))
            self.line_search.prepare_solve_call(values_spec=values_spec)

            deltas = TensorDict(((var.name, delta) for var, delta in zip(variables, deltas)))
            deltas = self.line_search.solve(
                x_init=deltas, base_value=loss_before, target_value=loss_step,
                estimated_improvement=estimated_improvement, fn_x=evaluate_step
            )

            return list(deltas.values())
