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

from tensorforce import util
from tensorforce.core import parameter_modules, TensorSpec, tf_function, tf_util
from tensorforce.core.optimizers import Optimizer
from tensorforce.core.optimizers.solvers import solver_modules


class NaturalGradient(Optimizer):
    """
    Natural gradient optimizer (specification key: `natural_gradient`).

    Args:
        learning_rate (parameter, float >= 0.0): Learning rate as KL-divergence of distributions
            between optimization steps (<span style="color:#C00000"><b>required</b></span>).
        cg_max_iterations (int >= 0): Maximum number of conjugate gradient iterations.
            (<span style="color:#00C000"><b>default</b></span>: 10).
        cg_damping (0.0 <= float <= 1.0): Conjugate gradient damping factor.
            (<span style="color:#00C000"><b>default</b></span>: 1e-3).
        cg_unroll_loop (bool): Whether to unroll the conjugate gradient loop
            (<span style="color:#00C000"><b>default</b></span>: false).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        optimized_module (module): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, learning_rate, cg_max_iterations=10, cg_damping=1e-3, cg_unroll_loop=False,
        summary_labels=None, name=None, arguments_spec=None, optimized_module=None
    ):
        super().__init__(
            summary_labels=summary_labels, name=name, arguments_spec=arguments_spec,
            optimized_module=optimized_module
        )

        self.learning_rate = self.add_module(
            name='learning_rate', module=learning_rate, modules=parameter_modules, dtype='float',
            min_value=0.0
        )

        self.conjugate_gradient = self.add_module(
            name='conjugate_gradient', module='conjugate_gradient', modules=solver_modules,
            max_iterations=cg_max_iterations, damping=cg_damping, unroll_loop=cg_unroll_loop,
            fn_values_spec=(lambda: [
                TensorSpec(type=tf_util.dtype(x=x), shape=tf_util.shape(x=x))
                for x in self.optimized_module.trainable_variables
            ])
        )

    @tf_function(num_args=1)
    def step(self, arguments, variables, fn_loss, **kwargs):
        # Optimize: argmin(w) loss(w + delta) such that kldiv(P(w) || P(w + delta)) = learning_rate
        # For more details, see our blogpost:
        # https://reinforce.io/blog/end-to-end-computation-graphs-for-reinforcement-learning/

        fn_kl_divergence = kwargs['fn_kl_divergence']
        return_estimated_improvement = kwargs.get('return_estimated_improvement', False)

        # Calculates the product x * F of a given vector x with the fisher matrix F.
        # Incorporating the product prevents having to calculate the entire matrix explicitly.
        def fisher_matrix_product(deltas):
            # Gradient is not propagated through solver.
            deltas = [tf.stop_gradient(input=delta) for delta in deltas]

            # kldiv
            kldiv = fn_kl_divergence(**arguments)

            # grad(kldiv)
            kldiv_grads = tf.gradients(ys=kldiv, xs=variables)
            kldiv_grads = [
                tf.zeros_like(input=var) if grad is None else grad
                for grad, var in zip(kldiv_grads, variables)
            ]

            # delta' * grad(kldiv)
            delta_kldiv_grads = tf.math.add_n(inputs=[
                tf.math.reduce_sum(input_tensor=(delta * grad))
                for delta, grad in zip(deltas, kldiv_grads)
            ])

            # [delta' * F] = grad(delta' * grad(kldiv))
            delta_kldiv_grads2 = tf.gradients(ys=delta_kldiv_grads, xs=variables)
            return [
                tf.zeros_like(input=var) if grad is None else grad
                for grad, var in zip(delta_kldiv_grads2, variables)
            ]

        # loss
        loss = fn_loss(**arguments)

        # grad(loss)
        loss_gradients = tf.gradients(ys=loss, xs=variables)
        loss_gradients = [
            tf.zeros_like(input=var) if grad is None else grad
            for var, grad in zip(variables, loss_gradients)
        ]

        # Solve the following system for delta' via the conjugate gradient solver.
        # [delta' * F] * delta' = -grad(loss)
        # --> delta'  (= lambda * delta)
        deltas = self.conjugate_gradient.solve(
            x_init=[tf.zeros_like(input=x) for x in variables], b=[-x for x in loss_gradients],
            fn_x=fisher_matrix_product
        )

        # delta' * F
        delta_fisher_matrix_product = fisher_matrix_product(deltas=deltas)

        # c' = 0.5 * delta' * F * delta'  (= lambda * c)
        # TODO: Why constant and hence KL-divergence sometimes negative?
        half = tf_util.constant(value=0.5, dtype='float')
        constant = half * tf.math.add_n(inputs=[
            tf.math.reduce_sum(input_tensor=(delta_F * delta))
            for delta_F, delta in zip(delta_fisher_matrix_product, deltas)
        ])

        learning_rate = self.learning_rate.value()

        # Zero step if constant <= 0
        def no_step():
            zero_deltas = [tf.zeros_like(input=delta) for delta in deltas]
            if return_estimated_improvement:
                return zero_deltas, tf_util.constant(value=0.0, dtype='float')
            else:
                return zero_deltas

        # Natural gradient step if constant > 0
        def apply_step():
            # lambda = sqrt(c' / c)
            lagrange_multiplier = tf.math.sqrt(x=(constant / learning_rate))

            # delta = delta' / lambda
            estimated_deltas = [delta / lagrange_multiplier for delta in deltas]

            # improvement = grad(loss) * delta  (= loss_new - loss_old)
            estimated_improvement = tf.math.add_n(inputs=[
                tf.math.reduce_sum(input_tensor=(grad * delta))
                for grad, delta in zip(loss_gradients, estimated_deltas)
            ])

            # Apply natural gradient improvement.
            assignments = list()
            for variable, delta in zip(variables, estimated_deltas):
                assignments.append(variable.assign_add(delta=delta, read_value=False))

            with tf.control_dependencies(control_inputs=assignments):
                # Trivial operation to enforce control dependency
                estimated_delta = util.fmap(function=tf_util.identity, xs=estimated_deltas)
                if return_estimated_improvement:
                    return estimated_delta, estimated_improvement
                else:
                    return estimated_delta

        # Natural gradient step only works if constant > 0
        skip_step = constant > tf_util.constant(value=0.0, dtype='float')
        return tf.cond(pred=skip_step, true_fn=no_step, false_fn=apply_step)
