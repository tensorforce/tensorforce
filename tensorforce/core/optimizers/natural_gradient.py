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
from tensorforce.core import parameter_modules, TensorDict, tf_function, tf_util
from tensorforce.core.optimizers import Optimizer
from tensorforce.core.optimizers.solvers import solver_modules


class NaturalGradient(Optimizer):
    """
    Natural gradient optimizer (specification key: `natural_gradient`).

    Args:
        learning_rate (parameter, float > 0.0): Learning rate as KL-divergence of distributions
            between optimization steps
            (<span style="color:#C00000"><b>required</b></span>).
        cg_max_iterations (int >= 1): Maximum number of conjugate gradient iterations.
            (<span style="color:#00C000"><b>default</b></span>: 10).
        cg_damping (0.0 <= float <= 1.0): Conjugate gradient damping factor.
            (<span style="color:#00C000"><b>default</b></span>: 0.1).
        only_positive_updates (bool): Whether to only perform updates with positive improvement
            estimate
            (<span style="color:#00C000"><b>default</b></span>: true).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, learning_rate, cg_max_iterations=10, cg_damping=0.1, only_positive_updates=True,
        name=None, arguments_spec=None
    ):
        super().__init__(name=name, arguments_spec=arguments_spec)

        self.learning_rate = self.submodule(
            name='learning_rate', module=learning_rate, modules=parameter_modules, dtype='float',
            min_value=0.0
        )

        self.conjugate_gradient = self.submodule(
            name='conjugate_gradient', module='conjugate_gradient', modules=solver_modules,
            max_iterations=cg_max_iterations, damping=cg_damping
        )

        self.only_positive_updates = only_positive_updates

    def initialize_given_variables(self, *, variables):
        super().initialize_given_variables(variables=variables)

        self.conjugate_gradient.complete_initialize(
            arguments_spec=self.arguments_spec, values_spec=self.variables_spec
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        # Optimize: argmin(w) loss(w + delta) such that kldiv(P(w) || P(w + delta)) = learning_rate
        # For more details, see our blogpost:
        # https://reinforce.io/blog/end-to-end-computation-graphs-for-reinforcement-learning/

        fn_kl_divergence = kwargs['fn_kl_divergence']

        # TODO: should be moved to initialize_given_variables, but fn_kl_divergence...
        # Calculates the product x * F of a given vector x with the fisher matrix F.
        # Incorporating the product prevents having to calculate the entire matrix explicitly.
        def fisher_matrix_product(arguments, deltas):
            # Second-order gradients
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape1:
                for variable in variables:
                    tape1.watch(tensor=variable)
                with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape2:
                    for variable in variables:
                        tape2.watch(tensor=variable)

                    # kldiv
                    kldiv = fn_kl_divergence(**arguments.to_kwargs())

                # grad(kldiv)
                kldiv_grads = tape2.gradient(target=kldiv, sources=variables)
                kldiv_grads = [
                    tf.zeros_like(input=var) if grad is None else grad
                    for var, grad in zip(variables, kldiv_grads)
                ]

                # delta' * grad(kldiv)
                multiply = functools.partial(
                    tf_util.lift_indexedslices, tf.math.multiply,
                    with_assertions=self.config.create_tf_assertions
                )
                delta_kldiv_grads = tf.math.add_n(inputs=[
                    tf.math.reduce_sum(input_tensor=multiply(delta, grad))
                    for delta, grad in zip(deltas.values(), kldiv_grads)
                ])

            # [delta' * F] = grad(delta' * grad(kldiv))
            delta_kldiv_grads2 = tape1.gradient(target=delta_kldiv_grads, sources=variables)
            return TensorDict((
                (var.name[:-2].replace('/', '_'), tf.zeros_like(input=var) if x is None else x)
                for var, x in zip(variables, delta_kldiv_grads2)
            ))

        # loss
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            for variable in variables:
                tape.watch(tensor=variable)
            loss = fn_loss(**arguments.to_kwargs())

        # grad(loss)
        loss_gradients = tape.gradient(target=loss, sources=variables)
        loss_gradients = [
            tf.zeros_like(input=var) if grad is None else grad
            for var, grad in zip(variables, loss_gradients)
        ]

        # Solve the following system for delta' via the conjugate gradient solver.
        # [delta' * F] * delta' = -grad(loss)
        # --> delta'  (= lambda * delta)
        # Replace "/" with "_" to ensure TensorDict is flat
        x_init = TensorDict(
            ((var.name[:-2].replace('/', '_'), tf.zeros_like(input=var)) for var in variables)
        )
        b = TensorDict(
            ((var.name[:-2].replace('/', '_'), -x) for var, x in zip(variables, loss_gradients))
        )
        deltas = self.conjugate_gradient.solve(
            arguments=arguments, x_init=x_init, b=b, fn_x=fisher_matrix_product
        )

        # delta' * F
        delta_fisher_matrix_product = fisher_matrix_product(arguments=arguments, deltas=deltas)

        # c' = 0.5 * delta' * F * delta'  (= lambda^2 * c)
        # TODO: Why constant and hence KL-divergence sometimes negative?
        delta_F_delta = delta_fisher_matrix_product.fmap(
            function=(lambda delta_F, delta: delta_F * delta), zip_values=deltas
        )
        half = tf_util.constant(value=0.5, dtype='float')
        constant = half * tf.math.add_n(inputs=[
            tf.math.reduce_sum(input_tensor=x) for x in delta_F_delta.values()
        ])

        learning_rate = self.learning_rate.value()
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        # Zero step if constant <= 0
        def no_step():
            return deltas.fmap(function=tf.zeros_like)

        # Natural gradient step if constant > 0
        def apply_step():
            # lambda = sqrt(c' / c)
            lagrange_multiplier = tf.where(
                condition=(constant > 0.0),
                x=tf.math.sqrt(x=(constant / tf.math.maximum(x=learning_rate, y=epsilon))),
                y=-tf.math.sqrt(x=(-constant / tf.math.maximum(x=learning_rate, y=epsilon)))
            )

            # delta = delta' / lambda
            lagrange_multiplier = tf.where(
                condition=(lagrange_multiplier > 0.0),
                x=tf.math.maximum(x=lagrange_multiplier, y=epsilon),
                y=tf.math.minimum(x=lagrange_multiplier, y=-epsilon)
            )
            estimated_deltas = deltas.fmap(function=(lambda delta: delta / lagrange_multiplier))

            # Apply natural gradient improvement.
            assignments = list()
            for variable, delta in zip(variables, estimated_deltas.values()):
                assignments.append(variable.assign_add(delta=delta, read_value=False))

            with tf.control_dependencies(control_inputs=assignments):
                # # improvement = grad(loss) * delta  (= loss_new - loss_old)
                # improvement_estimate = tf.math.add_n(inputs=[
                #     tf.math.reduce_sum(input_tensor=(loss_grad * delta))
                #     for loss_grad, delta in zip(loss_gradients, estimated_deltas.values())
                # ])
                return estimated_deltas.fmap(function=tf_util.identity)

        if self.only_positive_updates:
            # Natural gradient step only works if constant > 0 (epsilon to avoid zero division)
            skip_step = constant < (epsilon * learning_rate)
            return tf.cond(pred=skip_step, true_fn=no_step, false_fn=apply_step)

        else:
            return apply_step()
