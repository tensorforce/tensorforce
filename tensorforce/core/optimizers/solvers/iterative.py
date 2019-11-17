# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from tensorforce import util
from tensorforce.core import parameter_modules
from tensorforce.core.optimizers.solvers import Solver


class Iterative(Solver):
    """
    Generic solver which *iteratively* solves an equation/optimization problem. Involves an  
    initialization step, the iteration loop body and the termination condition.
    """

    def __init__(self, name, max_iterations, unroll_loop):
        """
        Creates a new iterative solver instance.

        Args:
            max_iterations: Maximum number of iterations before termination.
            unroll_loop: Unrolls the TensorFlow while loop if true.
        """
        super().__init__(name=name)

        assert isinstance(unroll_loop, bool)
        self.unroll_loop = unroll_loop

        if self.unroll_loop:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = self.add_module(
                name='max-iterations', module=max_iterations, modules=parameter_modules,
                dtype='int'
            )

    def tf_solve(self, fn_x, x_init, *args):
        """
        Iteratively solves an equation/optimization for $x$ involving an expression $f(x)$.

        Args:
            fn_x: A callable returning an expression $f(x)$ given $x$.
            x_init: Initial solution guess $x_0$.
            *args: Additional solver-specific arguments.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        self.fn_x = fn_x

        # Initialization step
        args = self.start(x_init, *args)

        # Iteration loop with termination condition
        if self.unroll_loop:
            # Unrolled for loop
            for _ in range(self.max_iterations):
                next_step = self.next_step(*args)
                step = (lambda: self.step(*args))
                do_nothing = (lambda: args)
                args = self.cond(pred=next_step, true_fn=step, false_fn=do_nothing)

        else:
            # TensorFlow while loop
            max_iterations = self.max_iterations.value()
            args = self.while_loop(
                cond=self.next_step, body=self.step, loop_vars=args, back_prop=False,
                maximum_iterations=max_iterations
            )

        solution = self.end(*args)

        return solution

    def tf_start(self, x_init, *args):
        """
        Initialization step preparing the arguments for the first iteration of the loop body.

        Args:
            x_init: Initial solution guess $x_0$.
            *args: Additional solver-specific arguments.

        Returns:
            Initial arguments for tf_step.
        """
        return (x_init,) + args

    def tf_step(self, x, *args):
        """
        Iteration loop body of the iterative solver.

        Args:
            x: Current solution estimate.
            *args: Additional solver-specific arguments.

        Returns:
            Updated arguments for next iteration.
        """
        raise NotImplementedError

    def tf_next_step(self, x, *args):
        """
        Termination condition (default: max number of iterations).

        Args:
            x: Current solution estimate.
            *args: Additional solver-specific arguments.

        Returns:
            True if another iteration should be performed.
        """
        return util.tf_always_true()

    def tf_end(self, x_final, *args):
        """
        Termination step preparing the return value.

        Args:
            x: Final solution estimate.
            *args: Additional solver-specific arguments.

        Returns:
            Final solution.
        """
        return x_final
