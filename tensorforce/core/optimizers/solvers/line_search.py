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

from tensorforce import util
from tensorforce.core import parameter_modules, SignatureDict, TensorSpec, tf_function, tf_util
from tensorforce.core.optimizers.solvers import Iterative


class LineSearch(Iterative):
    """
    Line search algorithm which iteratively optimizes the value $f(x)$ for $x$ on the line between
    $x'$ and $x_0$ by optimistically taking the first acceptable $x$ starting from $x_0$ and
    moving towards $x'$.
    """

    def __init__(self, *, name, max_iterations, backtracking_factor):
        """
        Create a new line search solver instance.

        Args:
            max_iterations (parameter, int >= 1): Maximum number of iterations before termination.
            backtracking_factor (parameter, 0.0 < float < 1.0): Backtracking factor.
        """
        super().__init__(name=name, max_iterations=max_iterations)

        self.backtracking_factor = self.submodule(
            name='backtracking_factor', module=backtracking_factor, modules=parameter_modules,
            dtype='float', min_value=0.0, max_value=1.0
        )

    def input_signature(self, *, function):
        if function == 'end' or function == 'next_step' or function == 'step':
            return SignatureDict(
                arguments=self.arguments_spec.signature(batched=True),
                x=self.values_spec.signature(batched=False),
                deltas=self.values_spec.signature(batched=False),
                improvement=TensorSpec(type='float', shape=()).signature(batched=False),
                last_improvement=TensorSpec(type='float', shape=()).signature(batched=False),
                base_value=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        elif function == 'solve' or function == 'start':
            return SignatureDict(
                arguments=self.arguments_spec.signature(batched=True),
                x_init=self.values_spec.signature(batched=False),
                base_value=TensorSpec(type='float', shape=()).signature(batched=False),
                zero_value=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'end' or function == 'solve':
            return SignatureDict(singleton=self.values_spec.signature(batched=False))

        elif function == 'next_step':
            return SignatureDict(
                singleton=TensorSpec(type='bool', shape=()).signature(batched=False)
            )

        elif function == 'start' or function == 'step':
            return SignatureDict(
                arguments=self.arguments_spec.signature(batched=True),
                x=self.values_spec.signature(batched=False),
                deltas=self.values_spec.signature(batched=False),
                improvement=TensorSpec(type='float', shape=()).signature(batched=False),
                last_improvement=TensorSpec(type='float', shape=()).signature(batched=False),
                base_value=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=4)
    def solve(self, *, arguments, x_init, base_value, zero_value, fn_x):
        """
        Iteratively optimize $f(x)$ for $x$ on the line between $x'$ and $x_0$.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            zero_value: Value $f(x_0)$ at $x = x_0$.
            fn_x: A callable returning the value $f(x)$ at $x$, with potential side effect.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        return super().solve(
            arguments=arguments, x_init=x_init, base_value=base_value, zero_value=zero_value,
            fn_x=fn_x
        )

    @tf_function(num_args=4)
    def start(self, *, arguments, x_init, base_value, zero_value):
        """
        Initialization step preparing the arguments for the first iteration of the loop body.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            zero_value: Value $f(x_0)$ at $x = x_0$.

        Returns:
            Initial arguments for step.
        """
        one_float = tf_util.constant(value=1.0, dtype='float')
        backtracking_factor = self.backtracking_factor.value()
        deltas = x_init.fmap(function=(lambda t: t * (backtracking_factor - one_float)))

        last_improvement = base_value - zero_value

        target_value = self.fn_x(arguments, deltas)
        improvement = base_value - target_value

        return arguments, x_init, deltas, improvement, last_improvement, base_value

    @tf_function(num_args=6, is_loop_body=True)
    def step(self, *, arguments, x, deltas, improvement, last_improvement, base_value):
        """
        Iteration loop body of the line search algorithm.

        Args:
            x: Current solution estimate $x_{t-1}$.
            deltas: Current difference $x_t - x_{t-1}$.
            improvement: Current improvement $(f(x') - f(x_t))$.
            last_improvement: Last improvement $(f(x') - f(x_{t-1}))$.
            base_value: Value $f(x')$ at $x = x'$.

        Returns:
            Updated arguments for next iteration.
        """
        next_x = x.fmap(function=(lambda t, delta: t + delta), zip_values=deltas)

        backtracking_factor = self.backtracking_factor.value()
        next_deltas = deltas.fmap(function=(lambda delta: delta * backtracking_factor))

        target_value = self.fn_x(arguments, next_deltas)
        next_improvement = base_value - target_value

        return arguments, next_x, next_deltas, next_improvement, improvement, base_value

    @tf_function(num_args=6)
    def next_step(self, *, arguments, x, deltas, improvement, last_improvement, base_value):
        """
        Termination condition: max number of iterations, or no improvement for last step, or
        improvement less than acceptable ratio, or estimated value not positive.

        Args:
            x: Current solution estimate $x_{t-1}$.
            deltas: Current difference $x_t - x_{t-1}$.
            improvement: Current improvement $(f(x') - f(x_t))$.
            last_improvement: Last improvement $(f(x') - f(x_{t-1}))$.
            base_value: Value $f(x')$ at $x = x'$.

        Returns:
            True if another iteration should be performed.
        """
        return improvement > last_improvement

    @tf_function(num_args=6)
    def end(self, *, arguments, x, deltas, improvement, last_improvement, base_value):
        """
        Termination step preparing the return value.

        Args:
            x: Final solution estimate $x_n$.
            deltas: Current difference $x_n - x_{n-1}$.
            improvement: Current improvement $(f(x') - f(x_t))$.
            last_improvement: Last improvement $(f(x') - f(x_{t-1}))$.
            base_value: Value $f(x')$ at $x = x'$.

        Returns:
            Final solution.
        """

        def keep_last_step():
            return x.fmap(function=(lambda t, delta: t + delta), zip_values=deltas)

        def undo_last_step():
            target_value = self.fn_x(arguments, deltas.fmap(function=(lambda delta: -delta)))

            dependencies = [target_value]
            if self.config.create_debug_assertions:
                epsilon = tf_util.constant(value=1e-5, dtype='float')
                epsilon = tf.math.maximum(x=epsilon, y=(epsilon * tf.math.abs(x=base_value)))
                dependencies.append(tf.debugging.assert_less(
                    x=tf.math.abs(x=(base_value - target_value - last_improvement)), y=epsilon
                ))

            with tf.control_dependencies(control_inputs=dependencies):
                return x.fmap(function=tf_util.identity)

        accept_solution = (improvement >= last_improvement)
        return tf.cond(pred=accept_solution, true_fn=keep_last_step, false_fn=undo_last_step)
