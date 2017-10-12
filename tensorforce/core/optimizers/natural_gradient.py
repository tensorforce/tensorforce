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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce.core.optimizers import Optimizer
from tensorforce.core.optimizers.solvers import ConjugateGradient


class NaturalGradient(Optimizer):
    """Natural gradient optimizer."""

    def __init__(self, learning_rate, cg_max_iterations=20, cg_damping=1e-3):
        super(NaturalGradient, self).__init__()

        assert learning_rate > 0.0
        self.kl_divergence = learning_rate

        self.solver = ConjugateGradient(max_iterations=cg_max_iterations, damping=cg_damping)

    def tf_step(self, time, variables, fn_loss, fn_kl_divergence, **kwargs):
        # Optimize: argmin(delta) loss(theta + delta) s.t. kldiv = c
        # Approximate:
        # - kldiv = 0.5 * delta^T * F * delta
        # - loss(theta + delta) = loss + grad(loss) * delta

        loss = fn_loss()
        loss_gradient = tf.gradients(ys=loss, xs=variables)  # grad(loss)
        kl_gradient = tf.gradients(ys=fn_kl_divergence(), xs=variables)  # grad(kl_div)

        # Approximate search direction
        def fisher_matrix_product(x):
            # Gradient is not propagated through solver
            x = [tf.stop_gradient(input=t) for t in x]

            # grad(kl_div) * x
            kl_gradient_x = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(grad * t)) for grad, t in zip(kl_gradient, x)])

            # F*x = grad(grad(kl_div) * x)
            return tf.gradients(ys=kl_gradient_x, xs=variables)

        fisher_matrix_product = tf.make_template(name_='fisher_matrix_product', func_=fisher_matrix_product)

        # [F*[delta*lambda]] * [delta*lambda] = -grad(loss)
        diffs = self.solver.solve(fn_x=fisher_matrix_product, x_init=None, b=[-grad for grad in loss_gradient])
        fisher = fisher_matrix_product(x=diffs)

        # [c*lambda^2] = 0.5 * [F*[delta*lambda]] * [delta*lambda]
        constant = 0.5 * tf.add_n(inputs=[tf.reduce_sum(input_tensor=(diff * f)) for diff, f in zip(diffs, fisher)])


        # why constant sometimes negative?
        # with tf.control_dependencies((tf.assert_greater(x=constant, y=0.0, message='constant <= epsilon!'),)):

        def true_fn():
            # lambda = sqrt([c*lambda^2] / c)
            lagrange_multiplier = tf.sqrt(x=(constant / self.kl_divergence))
            # [delta*lambda] / lambda
            estimated_diffs = [diff / lagrange_multiplier for diff in diffs]
            # deriv(loss)^T * sum(delta)
            estimated_improvement = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(grad * diff))
                                                     for grad, diff in zip(loss_gradient, estimated_diffs)])

            applied = self.apply_step(variables=variables, diffs=estimated_diffs)

            with tf.control_dependencies(control_inputs=(applied,)):
                return [tf.identity(input=estimated_diff) for estimated_diff in estimated_diffs]

        def false_fn():
            return [tf.zeros_like(tensor=diff) for diff in diffs]

        # NOTE: line search doesn't use estimated_improvement !!!

        return tf.cond(pred=(constant > 0.0), true_fn=true_fn, false_fn=false_fn)
