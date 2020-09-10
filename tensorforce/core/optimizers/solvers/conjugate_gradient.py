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

import functools

import tensorflow as tf

from tensorforce import util
from tensorforce.core import parameter_modules, SignatureDict, TensorSpec, tf_function, tf_util
from tensorforce.core.optimizers.solvers import Iterative


class ConjugateGradient(Iterative):
    """
    Conjugate gradient algorithm which iteratively finds a solution $x$ for a system of linear  
    equations of the form $A x = b$, where $A x$ could be, for instance, a locally linear  
    approximation of a high-dimensional function.

    See below pseudo-code taken from  
    [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm):

    ```text
    def conjgrad(A, b, x_0):
        r_0 := b - A * x_0
        c_0 := r_0
        r_0^2 := r^T * r

        for t in 0, ..., max_iterations - 1:
            Ac := A * c_t
            cAc := c_t^T * Ac
            \alpha := r_t^2 / cAc
            x_{t+1} := x_t + \alpha * c_t
            r_{t+1} := r_t - \alpha * Ac
            r_{t+1}^2 := r_{t+1}^T * r_{t+1}
            if r_{t+1} < \epsilon:
                break
            \beta = r_{t+1}^2 / r_t^2
            c_{t+1} := r_{t+1} + \beta * c_t

        return x_{t+1}
    ```

    """

    def __init__(self, *, name, max_iterations, damping):
        """
        Creates a new conjugate gradient solver instance.

        Args:
            max_iterations (parameter, int >= 1): Maximum number of iterations before termination.
            damping (parameter, 0.0 <= float <= 1.0): Damping factor.
        """
        super().__init__(name=name, max_iterations=max_iterations)

        self.damping = self.submodule(
            name='damping', module=damping, modules=parameter_modules, dtype='float', min_value=0.0,
            max_value=1.0
        )

    def input_signature(self, *, function):
        if function == 'end' or function == 'next_step' or function == 'step':
            return SignatureDict(
                arguments=self.arguments_spec.signature(batched=True),
                x=self.values_spec.signature(batched=False),
                conjugate=self.values_spec.signature(batched=False),
                residual=self.values_spec.signature(batched=False),
                squared_residual=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        elif function == 'solve' or function == 'start':
            return SignatureDict(
                arguments=self.arguments_spec.signature(batched=True),
                x_init=self.values_spec.signature(batched=False),
                b=self.values_spec.signature(batched=False)
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
                conjugate=self.values_spec.signature(batched=False),
                residual=self.values_spec.signature(batched=False),
                squared_residual=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=3)
    def solve(self, *, arguments, x_init, b, fn_x):
        """
        Iteratively solves the system of linear equations $A x = b$.

        Args:
            arguments: ???
            x_init: Initial solution guess $x_0$, zero vector if None.
            b: The right-hand side $b$ of the system of linear equations.
            fn_x: A callable returning the left-hand side $A x$ of the system of linear equations.

        Returns:
            A solution $x$ to the problem as given by the solver.
        """
        return super().solve(arguments=arguments, x_init=x_init, b=b, fn_x=fn_x)

    @tf_function(num_args=3)
    def start(self, *, arguments, x_init, b):
        """
        Initialization step preparing the arguments for the first iteration of the loop body:
        $x_0, 0, p_0, r_0, r_0^2$.

        Args:
            arguments: ???
            x_init: Initial solution guess $x_0$, zero vector if None.
            b: The right-hand side $b$ of the system of linear equations.

        Returns:
            Initial arguments for step.
        """
        # r_0 := b - A * x_0
        # c_0 := r_0
        fx = self.fn_x(arguments, x_init)
        subtract = functools.partial(
            tf_util.lift_indexedslices, tf.math.subtract,
            with_assertions=self.config.create_tf_assertions
        )
        conjugate = residual = b.fmap(function=subtract, zip_values=fx)

        # r_0^2 := r^T * r
        multiply = functools.partial(
            tf_util.lift_indexedslices, tf.math.multiply,
            with_assertions=self.config.create_tf_assertions
        )
        squared_residual = tf.math.add_n(inputs=[
            tf.math.reduce_sum(input_tensor=multiply(res, res)) for res in residual.values()
        ])

        return arguments, x_init, conjugate, residual, squared_residual

    @tf_function(num_args=5, is_loop_body=True)
    def step(self, *, arguments, x, conjugate, residual, squared_residual):
        """
        Iteration loop body of the conjugate gradient algorithm.

        Args:
            arguments: ???
            x: Current solution estimate $x_t$.
            conjugate: Current conjugate $c_t$.
            residual: Current residual $r_t$.
            squared_residual: Current squared residual $r_t^2$.

        Returns:
            Updated arguments for next iteration.
        """

        # Ac := A * c_t
        A_conjugate = self.fn_x(arguments, conjugate)

        # TODO: reference?
        damping = self.damping.value()

        def no_damping():
            return A_conjugate

        def apply_damping():
            return A_conjugate.fmap(
                function=(lambda A_conj, conj: A_conj + damping * conj), zip_values=conjugate
            )

        zero = tf_util.constant(value=0.0, dtype='float')
        skip_damping = tf.math.equal(x=damping, y=zero)
        A_conjugate = tf.cond(pred=skip_damping, true_fn=no_damping, false_fn=apply_damping)

        # cAc := c_t^T * Ac
        multiply = tf.math.multiply
        conjugate_A_conjugate = conjugate.fmap(function=multiply, zip_values=A_conjugate)
        conjugate_A_conjugate = tf.math.add_n(inputs=[
            tf.math.reduce_sum(input_tensor=conj_A_conj)
            for conj_A_conj in conjugate_A_conjugate.values()
        ])

        # \alpha := r_t^2 / cAc
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        conjugate_A_conjugate = tf.where(
            condition=(conjugate_A_conjugate > 0.0),
            x=tf.math.maximum(x=conjugate_A_conjugate, y=epsilon),
            y=tf.math.minimum(x=conjugate_A_conjugate, y=-epsilon)
        )
        alpha = squared_residual / conjugate_A_conjugate

        # x_{t+1} := x_t + \alpha * c_t
        next_x = x.fmap(function=(lambda t, conj: t + alpha * conj), zip_values=conjugate)

        # r_{t+1} := r_t - \alpha * Ac
        next_residual = residual.fmap(
            function=(lambda res, A_conj: res - alpha * A_conj), zip_values=A_conjugate
        )

        # r_{t+1}^2 := r_{t+1}^T * r_{t+1}
        next_squared_residual = tf.math.add_n(
            inputs=[tf.math.reduce_sum(input_tensor=(res * res)) for res in next_residual.values()]
        )

        # \beta = r_{t+1}^2 / r_t^2
        beta = next_squared_residual / tf.math.maximum(x=squared_residual, y=epsilon)

        # c_{t+1} := r_{t+1} + \beta * c_t
        next_conjugate = next_residual.fmap(
            function=(lambda res, conj: res + beta * conj), zip_values=conjugate
        )

        return arguments, next_x, next_conjugate, next_residual, next_squared_residual

    @tf_function(num_args=5)
    def next_step(self, *, arguments, x, conjugate, residual, squared_residual):
        """
        Termination condition: max number of iterations, or residual sufficiently small.

        Args:
            arguments: ???
            x: Current solution estimate $x_t$.
            conjugate: Current conjugate $c_t$.
            residual: Current residual $r_t$.
            squared_residual: Current squared residual $r_t^2$.

        Returns:
            True if another iteration should be performed.
        """
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        return squared_residual >= epsilon
