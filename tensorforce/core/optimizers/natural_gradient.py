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

from six.moves import xrange

import tensorflow as tf

from tensorforce.core.optimizers import Optimizer
from tensorforce.core.optimizers.solvers import ConjugateGradient, LineSearch


class NaturalGradient(Optimizer):
    """Natural gradient optimizer."""

    def __init__(self, learning_rate, variables=None, cg_max_iterations=20, cg_damping=1e-3):
        assert learning_rate > 0.0
        self.kl_divergence = learning_rate

        self.solver = ConjugateGradient(max_iterations=cg_max_iterations, damping=cg_damping)

        super(NaturalGradient, self).__init__(variables=variables)

    def tf_step(self, fn_loss, fn_kl_divergence, **kwargs):
        # Optimize: argmin(delta) loss(theta + delta) such that kldiv = c
        # Approximate:
        # - kldiv = 0.5 * delta^T * F * delta
        # - loss(theta + delta) = loss + grad(loss) * delta

        loss = fn_loss()
        loss_gradient = tf.gradients(ys=loss, xs=self.variables)  # grad(loss)
        kl_gradient = tf.gradients(ys=fn_kl_divergence(), xs=self.variables)  # grad(kl_div)

        # Approximate search direction
        def fisher_matrix_product(x):
            # Gradient is not propagated through solver
            x = [tf.stop_gradient(input=t) for t in x]

            # grad(kl_div) * x
            kl_gradient_x = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(grad * t)) for grad, t in zip(kl_gradient, x)])

            # F*x = grad(grad(kl_div) * x)
            return tf.gradients(ys=kl_gradient_x, xs=self.variables)

        fisher_matrix_product = tf.make_template(name_='fisher_matrix_product', func_=fisher_matrix_product)

        # [F*[delta*lambda]] * [delta*lambda] = -grad(loss)
        delta = self.solver.solve(fn_x=fisher_matrix_product, x_init=None, b=[-grad for grad in loss_gradient])
        fisher = fisher_matrix_product(x=delta)

        # [c*lambda^2] = 0.5 * [F*[delta*lambda]] * [delta*lambda]
        constant = 0.5 * tf.add_n(inputs=[tf.reduce_sum(input_tensor=(d * f)) for d, f in zip(delta, fisher)])


        # why constant sometimes negative?
        # with tf.control_dependencies((tf.assert_greater(x=constant, y=0.0, message='constant <= epsilon!'),)):

        def true_fn():
            # lambda = sqrt([c*lambda^2] / c)
            lagrange_multiplier = tf.sqrt(x=(constant / self.kl_divergence))
            # [delta*lambda] / lambda
            estimated_step = [d / lagrange_multiplier for d in delta]
            # deriv(loss)^T * sum(delta)
            estimated_improvement = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(grad * d)) for grad, d in zip(loss_gradient, estimated_step)])
            return self.apply_step(diffs=estimated_step), estimated_step

        def false_fn():
            return tf.no_op(), [tf.zeros_like(tensor=d) for d in delta]

        # NOTE: line search doesn't use estimated_improvement !!!

        return tf.cond(pred=(constant > 0.0), true_fn=true_fn, false_fn=false_fn)
