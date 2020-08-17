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

    def __init__(self, *, name, max_iterations, backtracking_factor, accept_ratio):
        """
        Create a new line search solver instance.

        Args:
            max_iterations (parameter, int >= 0): Maximum number of iterations before termination.
            backtracking_factor (parameter, 0.0 < float < 1.0): Backtracking factor.
            accept_ratio (parameter, 0.0 <= float <= 1.0): Lower limit of what improvement ratio
                over $x = x'$ is acceptable (based either on a given estimated improvement or with
                respect to the value at   $x = x'$).
        """
        super().__init__(name=name, max_iterations=max_iterations)

        self.backtracking_factor = self.submodule(
            name='backtracking_factor', module=backtracking_factor, modules=parameter_modules,
            dtype='float', min_value=0.0, max_value=1.0
        )

        assert accept_ratio >= 0.0
        self.accept_ratio = self.submodule(
            name='accept_ratio', module=accept_ratio, modules=parameter_modules, dtype='float',
            min_value=0.0, max_value=1.0
        )

    def input_signature(self, *, function):
        if function == 'end' or function == 'next_step' or function == 'step':
            return SignatureDict(
                arguments=self.arguments_spec.signature(batched=True),
                x=self.values_spec.signature(batched=False),
                deltas=self.values_spec.signature(batched=False),
                improvement=TensorSpec(type='float', shape=()).signature(batched=False),
                last_improvement=TensorSpec(type='float', shape=()).signature(batched=False),
                base_value=TensorSpec(type='float', shape=()).signature(batched=False),
                estimate=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        elif function == 'solve' or function == 'start':
            return SignatureDict(
                arguments=self.arguments_spec.signature(batched=True),
                x_init=self.values_spec.signature(batched=False),
                base_value=TensorSpec(type='float', shape=()).signature(batched=False),
                zero_value=TensorSpec(type='float', shape=()).signature(batched=False),
                estimate=TensorSpec(type='float', shape=()).signature(batched=False)
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
                base_value=TensorSpec(type='float', shape=()).signature(batched=False),
                estimate=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=5)
    def solve(self, *, arguments, x_init, base_value, zero_value, estimate, fn_x):
        """
        Iteratively optimize $f(x)$ for $x$ on the line between $x'$ and $x_0$.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            zero_value: Value $f(x_0)$ at $x = x_0$.
            estimate: Estimated improvement for $x = x_0$.
            fn_x: A callable returning the value $f(x)$ at $x$, with potential side effect.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        return super().solve(
            arguments=arguments, x_init=x_init, base_value=base_value, zero_value=zero_value,
            estimate=estimate, fn_x=fn_x
        )

    @tf_function(num_args=5)
    def start(self, *, arguments, x_init, base_value, zero_value, estimate):
        """
        Initialization step preparing the arguments for the first iteration of the loop body.

        Args:
            x_init: Initial solution guess $x_0$.
            base_value: Value $f(x')$ at $x = x'$.
            zero_value: Value $f(x_0)$ at $x = x_0$.
            estimate: Estimated value at $x = x_0$.

        Returns:
            Initial arguments for step.
        """
        # dependencies = list()
        # if self.config.create_tf_assertions:
        #     zero_float = tf_util.constant(value=0.0, dtype='float')
        #     dependencies.append(tf.debugging.assert_greater_equal(x=estimate, y=zero_float))

        # with tf.control_dependencies(control_inputs=dependencies):
        zeros_x = x_init.fmap(function=tf.zeros_like)

        improvement = zero_value - base_value
        last_improvement = tf_util.constant(value=-1.0, dtype='float')

        return arguments, zeros_x, x_init, improvement, last_improvement, base_value, estimate

    @tf_function(num_args=7, is_loop_body=True)
    def step(self, *, arguments, x, deltas, improvement, last_improvement, base_value, estimate):
        """
        Iteration loop body of the line search algorithm.

        Args:
            x: Current solution estimate $x_{t-1}$.
            deltas: Current difference $x_t - x_{t-1}$.
            improvement: Current improvement $(f(x_t) - f(x'))$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x'))$.
            base_value: Value $f(x')$ at $x = x'$.
            estimate: Current estimated value at $x_t$.

        Returns:
            Updated arguments for next iteration.
        """
        next_x = x.fmap(function=(lambda t, delta: t + delta), zip_values=deltas)

        backtracking_factor = self.backtracking_factor.value()
        next_estimate = estimate * backtracking_factor

        def first_iteration():
            one_float = tf_util.constant(value=1.0, dtype='float')
            return deltas.fmap(function=(lambda t: t * (backtracking_factor - one_float)))

        def other_iterations():
            return deltas.fmap(function=(lambda delta: delta * backtracking_factor))

        zero_float = tf_util.constant(value=0.0, dtype='float')
        next_deltas = tf.cond(
            pred=(last_improvement < zero_float), true_fn=first_iteration, false_fn=other_iterations
        )

        with tf.control_dependencies(control_inputs=(improvement,)):
            target_value = self.fn_x(arguments, next_deltas)
            next_improvement = target_value - base_value
            # target_value = tf.compat.v1.Print(target_value, ('round', next_improvement, improvement, target_value, base_value))

        with tf.control_dependencies(control_inputs=(target_value,)):
            next_deltas = next_deltas.fmap(function=tf_util.identity)

        return arguments, next_x, next_deltas, next_improvement, improvement, base_value, \
            next_estimate

    @tf_function(num_args=7)
    def next_step(
        self, *, arguments, x, deltas, improvement, last_improvement, base_value, estimate
    ):
        """
        Termination condition: max number of iterations, or no improvement for last step, or
        improvement less than acceptable ratio, or estimated value not positive.

        Args:
            x: Current solution estimate $x_{t-1}$.
            deltas: Current difference $x_t - x_{t-1}$.
            improvement: Current improvement $(f(x_t) - f(x'))$.
            last_improvement: Last improvement $(f(x_{t-1}) - f(x'))$.
            base_value: Value $f(x')$ at $x = x'$.
            estimate: Current estimated value at $x_t$.

        Returns:
            True if another iteration should be performed.
        """
        # Continue while current step is an improvement over last step
        zero_float = tf_util.constant(value=0.0, dtype='float')
        last_improvement = tf.math.maximum(x=last_improvement, y=zero_float)
        next_step = (improvement >= last_improvement)
        # Continue while estimated improvement is positive
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        next_step = tf.math.logical_and(x=next_step, y=(estimate > epsilon))
        # Continue while improvement ratio is below accept ratio, so not yet sufficient
        accept_ratio = self.accept_ratio.value()
        improvement_ratio = improvement / tf.math.maximum(x=estimate, y=epsilon)
        return tf.math.logical_and(x=next_step, y=(improvement_ratio < accept_ratio))

    @tf_function(num_args=7)
    def end(self, *, arguments, x, deltas, improvement, last_improvement, base_value, estimate):
        """
        Termination step preparing the return value.

        Args:
            x: Final solution estimate $x_n$.
            deltas: Current difference $x_n - x_{n-1}$.
            improvement: Current improvement $(f(x_n) - f(x'))$.
            last_improvement: Last improvement $(f(x_{n-1}) - f(x'))$.
            base_value: Value $f(x')$ at $x = x'$.
            estimate: Final estimated value at $x_n$.

        Returns:
            Final solution.
        """
        zero_float = tf_util.constant(value=0.0, dtype='float')
        last_improvement = tf.math.maximum(x=last_improvement, y=zero_float)

        def keep_last_step():
            return x.fmap(function=(lambda t, delta: t + delta), zip_values=deltas)

        def undo_last_step():
            target_value = self.fn_x(arguments, deltas.fmap(function=(lambda delta: -delta)))

            dependencies = [target_value]
            if self.config.create_debug_assertions:
                epsilon = tf_util.constant(value=util.epsilon, dtype='float')
                epsilon = tf.math.maximum(x=epsilon, y=(epsilon * tf.math.abs(x=base_value)))
                # target_value = tf.compat.v1.Print(target_value, (target_value - base_value - last_improvement, target_value, base_value, improvement, last_improvement, epsilon))
                dependencies.append(tf.debugging.assert_less(
                    x=tf.math.abs(x=(target_value - base_value - last_improvement)), y=epsilon
                ))

            with tf.control_dependencies(control_inputs=dependencies):
                return x.fmap(function=tf_util.identity)

        accept_solution = (improvement >= last_improvement)
        return tf.cond(pred=accept_solution, true_fn=keep_last_step, false_fn=undo_last_step)
