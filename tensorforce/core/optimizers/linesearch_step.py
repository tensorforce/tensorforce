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

    def initialize_given_variables(self, *, variables, register_summaries):
        super().initialize_given_variables(
            variables=variables, register_summaries=register_summaries
        )

        values_spec = TensorsSpec((
            (var.name, TensorSpec(type=tf_util.dtype(x=var), shape=tf_util.shape(x=var)))
            for var in variables
        ))
        self.line_search.complete_initialize(
            arguments_spec=self.arguments_spec, values_spec=values_spec
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

        with tf.control_dependencies(control_inputs=deltas):
            # Negative value since line search maximizes.
            loss_after = -fn_loss(**arguments.to_kwargs())

            if isinstance(deltas, tuple):
                # If 'return_estimated_improvement' argument exists.
                if len(deltas) != 2:
                    raise TensorforceError(message="Unexpected output of internal optimizer.")
                deltas, estimated_improvement = deltas
                # Negative value since line search maximizes.
                estimated_improvement = -estimated_improvement
            else:
                # Some big value
                estimated_improvement = tf.math.maximum(
                    x=tf.math.abs(x=(loss_after - loss_before)),
                    y=tf.math.maximum(x=loss_after, y=tf_util.constant(value=1.0, dtype='float'))
                ) * tf_util.constant(value=1000.0, dtype='float')

            # TODO: debug assertion
            dependencies = [loss_after]
            if self.config.create_debug_assertions:
                dependencies.append(tf.debugging.assert_none_equal(x=loss_before, y=loss_after))

        with tf.control_dependencies(control_inputs=dependencies):

            # TODO: should be moved to initialize_given_variables, but fn_loss...
            def evaluate_step(arguments, deltas):
                assignments = list()
                for variable, delta in zip(variables, deltas.values()):
                    assignments.append(variable.assign_add(delta=delta, read_value=False))
                with tf.control_dependencies(control_inputs=assignments):
                    # Negative value since line search maximizes.
                    return -fn_loss(**arguments.to_kwargs())

            deltas = TensorDict(((var.name, delta) for var, delta in zip(variables, deltas)))
            deltas = self.line_search.solve(
                arguments=arguments, x_init=deltas, base_value=loss_before, zero_value=loss_after,
                estimated=estimated_improvement, fn_x=evaluate_step
            )

            return list(deltas.values())
