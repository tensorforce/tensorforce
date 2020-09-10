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

from tensorforce.core import parameter_modules, tf_util
from tensorforce.core.optimizers.solvers import Solver


class Iterative(Solver):
    """
    Generic solver which *iteratively* solves an equation/optimization problem. Involves an  
    initialization step, the iteration loop body and the termination condition.
    """

    def __init__(self, *, name, max_iterations):
        """
        Creates a new iterative solver instance.

        Args:
            max_iterations (parameter, int >= 1): Maximum number of iterations before termination.
        """
        super().__init__(name=name)

        self.max_iterations = self.submodule(
            name='max_iterations', module=max_iterations, modules=parameter_modules,
            dtype='int', min_value=1
        )

    def complete_initialize(self, arguments_spec, values_spec):
        self.arguments_spec = arguments_spec
        self.values_spec = values_spec

    def solve(self, *, arguments, x_init, fn_x=None, **kwargs):
        """
        Iteratively solves an equation/optimization for $x$ involving an expression $f(x)$.

        Args:
            arguments: ???
            x_init: Initial solution guess $x_0$.
            fn_x: A callable returning an expression $f(x)$ given $x$.
            **values: Additional solver-specific arguments.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        self.fn_x = fn_x

        # Initialization step
        values = self.start(arguments=arguments, x_init=x_init, **kwargs)

        # Iteration loop with termination condition
        max_iterations = self.max_iterations.value()
        signature = self.input_signature(function='step')
        values = signature.kwargs_to_args(kwargs=values)
        values = tf.while_loop(
            cond=self.next_step, body=self.step, loop_vars=tuple(values),
            maximum_iterations=tf_util.int32(x=max_iterations)
        )
        values = signature.args_to_kwargs(args=values)
        solution = self.end(**values.to_kwargs())

        return solution

    def start(self, *, arguments, x_init, **kwargs):
        """
        Initialization step preparing the arguments for the first iteration of the loop body.

        Args:
            arguments: ???
            x_init: Initial solution guess $x_0$.
            *args: Additional solver-specific arguments.

        Returns:
            Initial arguments for step.
        """
        return (arguments, x_init) + tuple(kwargs.values())

    def step(self, *, arguments, x, **kwargs):
        """
        Iteration loop body of the iterative solver.

        Args:
            arguments: ???
            x: Current solution estimate.
            *args: Additional solver-specific arguments.

        Returns:
            Updated arguments for next iteration.
        """
        raise NotImplementedError

    def next_step(self, *, arguments, x, **kwargs):
        """
        Termination condition (default: max number of iterations).

        Args:
            arguments: ???
            x: Current solution estimate.
            *args: Additional solver-specific arguments.

        Returns:
            True if another iteration should be performed.
        """
        return tf_util.constant(value=True, dtype='bool')

    def end(self, *, arguments, x, **kwargs):
        """
        Termination step preparing the return value.

        Args:
            arguments: ???
            x: Final solution estimate.
            *args: Additional solver-specific arguments.

        Returns:
            Final solution.
        """
        return x
