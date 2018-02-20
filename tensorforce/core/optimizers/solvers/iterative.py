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

from tensorforce import util
from tensorforce.core.optimizers.solvers import Solver


class Iterative(Solver):
    """
    Generic solver which *iteratively* solves an equation/optimization problem. Involves an  
    initialization step, the iteration loop body and the termination condition.
    """

    def __init__(self, max_iterations, unroll_loop=False):
        """
        Creates a new iterative solver instance.

        Args:
            max_iterations: Maximum number of iterations before termination.
            unroll_loop: Unrolls the TensorFlow while loop if true.
        """
        assert max_iterations >= 0
        self.max_iterations = max_iterations

        assert isinstance(unroll_loop, bool)
        self.unroll_loop = unroll_loop

        super(Iterative, self).__init__()

        # TensorFlow functions
        self.initialize = tf.make_template(name_='initialize', func_=self.tf_initialize)
        self.step = tf.make_template(name_='step', func_=self.tf_step)
        self.next_step = tf.make_template(name_='next-step', func_=self.tf_next_step)

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
        args = self.initialize(x_init, *args)
        # args = util.map_tensors(fn=tf.stop_gradient, tensors=args)

        # Iteration loop with termination condition
        if self.unroll_loop:
            # Unrolled for loop
            for _ in range(self.max_iterations):
                next_step = self.next_step(*args)
                step = (lambda: self.step(*args))
                do_nothing = (lambda: args)
                args = tf.cond(pred=next_step, true_fn=step, false_fn=do_nothing)

        else:
            # TensorFlow while loop
            args = tf.while_loop(cond=self.next_step, body=self.step, loop_vars=args)

        # First argument contains solution
        return args[0]

    def tf_initialize(self, x_init, *args):
        """
        Initialization step preparing the arguments for the first iteration of the loop body  
        (default: initial solution guess and iteration counter).

        Args:
            x_init: Initial solution guess $x_0$.
            *args: Additional solver-specific arguments.

        Returns:
            Initial arguments for tf_step.
        """
        return x_init, 0

    def tf_step(self, x, iteration, *args):
        """
        Iteration loop body of the iterative solver (default: increment iteration step). The  
        first two loop arguments have to be the current solution estimate and the iteration step.

        Args:
            x: Current solution estimate.
            iteration: Current iteration counter.
            *args: Additional solver-specific arguments.

        Returns:
            Updated arguments for next iteration.
        """
        return (x, iteration + 1) + args

    def tf_next_step(self, x, iteration, *args):
        """
        Termination condition (default: max number of iterations).

        Args:
            x: Current solution estimate.
            iteration: Current iteration counter.
            *args: Additional solver-specific arguments.

        Returns:
            True if another iteration should be performed.
        """
        return iteration < self.max_iterations
