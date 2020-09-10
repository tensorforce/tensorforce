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
    Line-search-step update modifier, which performs a line search on the update step returned by
    the given optimizer to find a potentially superior smaller step size
    (specification key: `linesearch_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        max_iterations (parameter, int >= 1): Maximum number of line search iterations
            (<span style="color:#C00000"><b>required</b></span>).
        backtracking_factor (parameter, 0.0 < float < 1.0): Line search backtracking factor
            (<span style="color:#00C000"><b>default</b></span>: 0.75).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, optimizer, max_iterations, backtracking_factor=0.75, name=None, arguments_spec=None
    ):
        super().__init__(optimizer=optimizer, name=name, arguments_spec=arguments_spec)

        self.line_search = self.submodule(
            name='line_search', module='line_search', modules=solver_modules,
            max_iterations=max_iterations, backtracking_factor=backtracking_factor
        )

    def initialize_given_variables(self, *, variables):
        super().initialize_given_variables(variables=variables)

        self.line_search.complete_initialize(
            arguments_spec=self.arguments_spec, values_spec=self.variables_spec
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        loss_before = fn_loss(**arguments.to_kwargs())

        with tf.control_dependencies(control_inputs=(loss_before,)):
            deltas = self.optimizer.step(
                arguments=arguments, variables=variables, fn_loss=fn_loss, **kwargs
            )

        with tf.control_dependencies(control_inputs=deltas):

            def linesearch():
                loss_after = fn_loss(**arguments.to_kwargs())

                with tf.control_dependencies(control_inputs=(loss_after,)):
                    # Replace "/" with "_" to ensure TensorDict is flat
                    _deltas = TensorDict((
                        (var.name[:-2].replace('/', '_'), delta)
                        for var, delta in zip(variables, deltas)
                    ))

                    # TODO: should be moved to initialize_given_variables, but fn_loss...
                    def evaluate_step(arguments, deltas):
                        assignments = list()
                        for variable, delta in zip(variables, deltas.values()):
                            assignments.append(variable.assign_add(delta=delta, read_value=False))
                        with tf.control_dependencies(control_inputs=assignments):
                            return fn_loss(**arguments.to_kwargs())

                    return self.line_search.solve(
                        arguments=arguments, x_init=_deltas, base_value=loss_before,
                        zero_value=loss_after, fn_x=evaluate_step
                    )

            num_nonzero = list()
            for delta in deltas:
                num_nonzero.append(tf.math.count_nonzero(input=delta))
            num_nonzero = tf.math.add_n(inputs=num_nonzero)

            return tf.cond(pred=(num_nonzero == 0), true_fn=(lambda: deltas), false_fn=linesearch)
