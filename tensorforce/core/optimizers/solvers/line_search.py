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
    Line search algorithm which iteratively optimizes the value $f(x)$ for $x$ on the line between  
    $x'$ and $x_0$ by optimistically taking the first acceptable $x$ starting from $x_0$ and  
    moving towards $0$.
    """

    def __init__(self, max_iterations, accept_ratio, mode, parameter, unroll_loop=False):
        """
        Creates a new line search solver instance.

        Args:
            max_iterations: Maximum number of iterations before termination.
            accept_ratio: Lower limit of what improvement ratio over $x = x'$ is acceptable.
            mode: Mode of movement between $x_0$ and $x'$, either 'linear' or 'exponential'.
            parameter: Movement mode parameter, additive or multiplicative, respectively.
            unroll_loop: Unrolls the TensorFlow while loop if true.
        """
        assert accept_ratio >= 0.0
        self.accept_ratio = accept_ratio

        # TODO: Implement such sequences more generally, also useful for learning rate decay or so.
        if mode not in ('linear', 'exponential'):
            raise TensorForceError("Invalid line search mode: {}, please choose one of'linear' or 'exponential'".format(mode))
        self.mode = mode
        self.parameter = parameter

        super(LineSearch, self).__init__(max_iterations=max_iterations, unroll_loop=unroll_loop)

    def tf_solve(self, fn_x, x_init, base_value, target_value, estimated_value=None):
        """
        Iteratively optimizes $f(x)$ for $x$ on the line between $x'$ and $x_0$.

        Args:
            fn_x: A callable returning the value $f(x)$ at $x$.
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            target_value: Value $f(x_0)$ at $x = x_0$.
            estimated_value: Estimated value at $x = x_0$, $f(x')$ if None.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        return super(LineSearch, self).tf_solve(fn_x, x_init, base_value, target_value, estimated_value)

    def tf_initialize(self, x_init, base_value, target_value, estimated_value):
        """
        Initialization step preparing the arguments for the first iteration of the loop body.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            target_value: Value $f(x_0)$ at $x = x_0$.
            estimated_value: Estimated value at $x = x_0$, $f(x')$ if None.

        Returns:
            Initial arguments for tf_step.
        """
        self.base_value = base_value

        if estimated_value is None:
            estimated_value = base_value

        first_step = super(LineSearch, self).tf_initialize(x_init)

        improvement = tf.divide(
            x=(target_value - base_value),
            y=tf.maximum(x=estimated_value, y=util.epsilon)
        )

        last_improvement = improvement - 1.0

        if self.mode == 'linear':
            deltas = [-t * self.parameter for t in x_init]
            # self.x_incr = [t * self.parameter for t in x_init]
            self.estimated_incr = -estimated_value * self.parameter
            # next_x = [t + incr for t, incr in zip(x_init, self.x_incr)]
            estimated_value += self.estimated_incr

        elif self.mode == 'exponential':
            deltas = [-t * self.parameter for t in x_init]
            estimated_value *= self.parameter

        return first_step + (deltas, improvement, last_improvement, estimated_value)

    def tf_step(self, x, iteration, deltas, improvement, last_improvement, estimated_value):
        """
        Iteration loop body of the line search algorithm.

        Args:
            x: Current solution estimate $x_t$.
            iteration: Current iteration counter $t$.
            deltas: Current difference $x_t - x'$.
            improvement: Current improvement $(f(x_t) - f(x')) / v'$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x')) / v'$.
            estimated_value: Current estimated value $v'$.

        Returns:
            Updated arguments for next iteration.
        """
        x, next_iteration, deltas, improvement, last_improvement, estimated_value = super(LineSearch, self).tf_step(
            x, iteration, deltas, improvement, last_improvement, estimated_value
        )

        next_x = [t + delta for t, delta in zip(x, deltas)]

        if self.mode == 'linear':
            next_deltas = deltas
            next_estimated_value = estimated_value + self.estimated_incr

        elif self.mode == 'exponential':
            next_deltas = [delta * self.parameter for delta in deltas]
            next_estimated_value = estimated_value * self.parameter

        next_improvement = tf.divide(
            x=(self.fn_x(x=next_deltas) - self.base_value),
            y=tf.maximum(x=next_estimated_value, y=util.epsilon)
        )

        return next_x, next_iteration, next_deltas, next_improvement, improvement, next_estimated_value

    def tf_next_step(self, x, iteration, deltas, improvement, last_improvement, estimated_value):
        """
        Termination condition: max number of iterations, or no improvement for last step, or  
        improvement less than acceptable ratio, or estimated value not positive.

        Args:
            x: Current solution estimate $x_t$.
            iteration: Current iteration counter $t$.
            deltas: Current difference $x_t - x'$.
            improvement: Current improvement $(f(x_t) - f(x')) / v'$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x')) / v'$.
            estimated_value: Current estimated value $v'$.

        Returns:
            True if another iteration should be performed.
        """
        next_step = super(LineSearch, self).tf_next_step(
            x, iteration, deltas, improvement, last_improvement, estimated_value
        )

        def false_fn():
            value = self.fn_x(x=[-delta for delta in deltas])
            with tf.control_dependencies(control_inputs=(value,)):
                return False

        improved = tf.cond(pred=(improvement > last_improvement), true_fn=(lambda: True), false_fn=false_fn)

        next_step = tf.logical_and(x=next_step, y=improved)
        next_step = tf.logical_and(x=next_step, y=(improvement < self.accept_ratio))
        return tf.logical_and(x=next_step, y=(estimated_value > util.epsilon))
