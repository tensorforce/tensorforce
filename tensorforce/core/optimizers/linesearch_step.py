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


class LinesearchStep(UpdateModifier):
    """
    Line-search-step update modifier, which applies line search to the given optimizer to find a
    more optimal step size (specification key: `linesearch_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        max_iterations (parameter, int >= 0): Maximum number of line search iterations
            (<span style="color:#00C000"><b>default</b></span>: 10).
        backtracking_factor (parameter, 0.0 < float < 1.0): Line search backtracking factor
            (<span style="color:#00C000"><b>default</b></span>: 0.75).
        accept_ratio (parameter, 0.0 <= float <= 1.0): Line search acceptance ratio, not applicable
            in most situations (<span style="color:#00C000"><b>default</b></span>: 0.9).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, optimizer, max_iterations=10, backtracking_factor=0.75, accept_ratio=0.9,
        name=None, arguments_spec=None
    ):
        super().__init__(optimizer=optimizer, name=name, arguments_spec=arguments_spec)

        self.line_search = self.submodule(
            name='line_search', module='line_search', modules=solver_modules,
            max_iterations=max_iterations, backtracking_factor=backtracking_factor,
            accept_ratio=accept_ratio
        )

    def initialize_given_variables(self, *, variables):
        super().initialize_given_variables(variables=variables)

        self.line_search.complete_initialize(
            arguments_spec=self.arguments_spec, values_spec=self.variables_spec
        )

    @tf_function(num_args=1, is_loop_body=True)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        # Negative value since line search maximizes
        loss_before = -fn_loss(**arguments.to_kwargs())

        with tf.control_dependencies(control_inputs=(loss_before,)):
            deltas = self.optimizer.step(
                arguments=arguments, variables=variables, fn_loss=fn_loss, **kwargs
            )
            if isinstance(deltas, tuple) and len(deltas) == 2 and isinstance(deltas[0], TensorDict):
                # Negative value since line search maximizes
                improvement_estimate = -deltas[1]
                deltas = deltas[0]
            else:
                # Replace "/" with "_" to ensure TensorDict is flat
                deltas = TensorDict((
                    (var.name[:-2].replace('/', '_'), delta)
                    for var, delta in zip(variables, deltas)
                ))
                improvement_estimate = None

        with tf.control_dependencies(control_inputs=deltas):
            # Negative value since line search maximizes.
            loss_after = -fn_loss(**arguments.to_kwargs())

            if improvement_estimate is None:
                # Some big value
                improvement_estimate = tf.math.maximum(
                    x=tf.math.abs(x=(loss_after - loss_before)),
                    y=tf.math.maximum(x=loss_after, y=tf_util.constant(value=1.0, dtype='float'))
                ) * tf_util.constant(value=1000.0, dtype='float')

        with tf.control_dependencies(control_inputs=(loss_after, improvement_estimate)):

            # TODO: should be moved to initialize_given_variables, but fn_loss...
            def evaluate_step(arguments, deltas):
                assignments = list()
                for variable, delta in zip(variables, deltas.values()):
                    assignments.append(variable.assign_add(delta=delta, read_value=False))
                with tf.control_dependencies(control_inputs=assignments):
                    # Negative value since line search maximizes.
                    return -fn_loss(**arguments.to_kwargs())

            deltas = self.line_search.solve(
                arguments=arguments, x_init=deltas, base_value=loss_before, zero_value=loss_after,
                estimate=improvement_estimate, fn_x=evaluate_step
            )

            return list(deltas.values())
