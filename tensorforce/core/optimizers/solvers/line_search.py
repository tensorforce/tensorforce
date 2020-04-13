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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import parameter_modules, TensorSpec, tf_function, tf_util
from tensorforce.core.optimizers.solvers import Iterative


class LineSearch(Iterative):
    """
    Line search algorithm which iteratively optimizes the value $f(x)$ for $x$ on the line between  
    $x'$ and $x_0$ by optimistically taking the first acceptable $x$ starting from $x_0$ and  
    moving towards $x'$.
    """

    def __init__(
        self, name, max_iterations, accept_ratio, mode, parameter, unroll_loop=False,
        fn_values_spec=None
    ):
        """
        Creates a new line search solver instance.

        Args:
            max_iterations (parameter, int >= 0): Maximum number of iterations before termination.
            accept_ratio (parameter, 0.0 <= float <= 1.0): Lower limit of what improvement ratio
                over $x = x'$ is acceptable (based either on a given estimated improvement or with
                respect to the value at   $x = x'$).
            mode: Mode of movement between $x_0$ and $x'$, either 'linear' or 'exponential'.
            parameter (parameter, 0.0 <= float <= 1.0): Movement mode parameter, additive or
                multiplicative, respectively.
            unroll_loop: Unrolls the TensorFlow while loop if true.
        """
        super().__init__(name=name, max_iterations=max_iterations, unroll_loop=unroll_loop)

        assert accept_ratio >= 0.0
        self.accept_ratio = self.add_module(
            name='accept_ratio', module=accept_ratio, modules=parameter_modules, dtype='float',
            min_value=0.0, max_value=1.0
        )

        # TODO: Implement such sequences more generally, also useful for learning rate decay or so.
        if mode not in ('linear', 'exponential'):
            raise TensorforceError(
                "Invalid line search mode: {}, please choose one of 'linear' or 'exponential'".format(mode)
            )
        self.mode = mode

        self.parameter = self.add_module(
            name='parameter', module=parameter, modules=parameter_modules, dtype='float',
            min_value=0.0, max_value=1.0
        )

        self.fn_values_spec = fn_values_spec

    def input_signature(self, function):
        if function == 'end' or function == 'next_step' or function == 'step':
            if self.mode == 'linear':
                return [
                    self.fn_values_spec().signature(batched=False),
                    self.fn_values_spec().signature(batched=False),
                    TensorSpec(type='float', shape=()).signature(batched=False),
                    TensorSpec(type='float', shape=()).signature(batched=False),
                    TensorSpec(type='float', shape=()).signature(batched=False),
                    [
                        TensorSpec(type='float', shape=()).signature(batched=False),
                        TensorSpec(type='float', shape=()).signature(batched=False)
                    ]
                ]
            else:
                return [
                    self.fn_values_spec().signature(batched=False),
                    self.fn_values_spec().signature(batched=False),
                    TensorSpec(type='float', shape=()).signature(batched=False),
                    TensorSpec(type='float', shape=()).signature(batched=False),
                    TensorSpec(type='float', shape=()).signature(batched=False),
                    TensorSpec(type='float', shape=()).signature(batched=False)
                ]

        elif function == 'solve' or function == 'start':
            return [
                self.fn_values_spec().signature(batched=False),
                TensorSpec(type='float', shape=()).signature(batched=False),
                TensorSpec(type='float', shape=()).signature(batched=False),
                TensorSpec(type='float', shape=()).signature(batched=False)
            ]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=4)
    def solve(self, x_init, base_value, target_value, estimated_improvement, fn_x):
        """
        Iteratively optimizes $f(x)$ for $x$ on the line between $x'$ and $x_0$.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            target_value: Value $f(x_0)$ at $x = x_0$.
            estimated_improvement: Estimated improvement for $x = x_0$, $f(x')$ if None.
            fn_x: A callable returning the value $f(x)$ at $x$.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        return super().solve(
            x_init=x_init, base_value=base_value, target_value=target_value,
            estimated_improvement=estimated_improvement, fn_x=fn_x
        )

    @tf_function(num_args=4)
    def start(self, x_init, base_value, target_value, estimated_improvement):
        """
        Initialization step preparing the arguments for the first iteration of the loop body.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            target_value: Value $f(x_0)$ at $x = x_0$.
            estimated_improvement: Estimated value at $x = x_0$, $f(x')$ if None.

        Returns:
            Initial arguments for step.
        """
        difference = target_value - base_value
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        improvement = difference / tf.math.maximum(x=estimated_improvement, y=epsilon)

        last_improvement = improvement - 1.0
        parameter = self.parameter.value()

        if self.mode == 'linear':
            deltas = [-t * parameter for t in x_init]
            estimated_incr = -estimated_improvement * parameter
            additional = (base_value, estimated_incr)

        elif self.mode == 'exponential':
            deltas = [-t * parameter for t in x_init]
            additional = base_value

        return x_init, deltas, improvement, last_improvement, estimated_improvement, additional

    @tf_function(num_args=6)
    def step(self, x, deltas, improvement, last_improvement, estimated_improvement, additional):
        """
        Iteration loop body of the line search algorithm.

        Args:
            x: Current solution estimate $x_t$.
            deltas: Current difference $x_t - x'$.
            improvement: Current improvement $(f(x_t) - f(x')) / v'$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x')) / v'$.
            estimated_improvement: Current estimated value $v'$.
            additional: ...

        Returns:
            Updated arguments for next iteration.
        """
        next_x = [t + delta for t, delta in zip(x, deltas)]
        parameter = self.parameter.value()

        if self.mode == 'linear':
            base_value, estimated_incr = additional
            next_deltas = deltas
            next_estimated_improvement = estimated_improvement + estimated_incr

        elif self.mode == 'exponential':
            base_value = additional
            next_deltas = [delta * parameter for delta in deltas]
            next_estimated_improvement = estimated_improvement * parameter

        target_value = self.fn_x(next_deltas)

        difference = target_value - base_value
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        next_improvement = difference / tf.math.maximum(x=next_estimated_improvement, y=epsilon)

        return next_x, next_deltas, next_improvement, improvement, next_estimated_improvement, \
            additional

    @tf_function(num_args=6)
    def next_step(
        self, x, deltas, improvement, last_improvement, estimated_improvement, additional
    ):
        """
        Termination condition: max number of iterations, or no improvement for last step, or
        improvement less than acceptable ratio, or estimated value not positive.

        Args:
            x: Current solution estimate $x_t$.
            deltas: Current difference $x_t - x'$.
            improvement: Current improvement $(f(x_t) - f(x')) / v'$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x')) / v'$.
            estimated_improvement: Current estimated value $v'$.
            additional: ...

        Returns:
            True if another iteration should be performed.
        """
        improved = improvement > last_improvement
        accept_ratio = self.accept_ratio.value()
        next_step = tf.math.logical_and(x=improved, y=(improvement < accept_ratio))
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        return tf.math.logical_and(x=next_step, y=(estimated_improvement > epsilon))

    @tf_function(num_args=6)
    def end(
        self, x_final, deltas, improvement, last_improvement, estimated_improvement, additional
    ):
        """
        Termination step preparing the return value.

        Args:
            x_init: Final solution estimate $x_n$.
            deltas: Current difference $x_n - x'$.
            improvement: Current improvement $(f(x_n) - f(x')) / v'$.
            last_improvement: Last improvement $(f(x_{n-1}) - f(x')) / v'$.
            estimated_improvement: Current estimated value $v'$.
            additional: ...

        Returns:
            Final solution.
        """
        def accept_deltas():
            return [t + delta for t, delta in zip(x_final, deltas)]

        def undo_deltas():
            value = self.fn_x([-delta for delta in deltas])
            with tf.control_dependencies(control_inputs=(value,)):
                return util.fmap(function=tf_util.identity, xs=x_final)

        skip_undo_deltas = improvement > last_improvement
        x_final = tf.cond(pred=skip_undo_deltas, true_fn=accept_deltas, false_fn=undo_deltas)
        return x_final
