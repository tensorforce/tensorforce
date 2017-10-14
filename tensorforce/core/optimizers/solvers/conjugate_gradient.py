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
from tensorforce.core.optimizers.solvers import Iterative


class ConjugateGradient(Iterative):
    """
    Iteratively finds a solution x for linear systems of the form Ax = b,
    where Ax typically is a local linear approximation of a high-dimensional function.


       Wikipedia: Conjugate Gradient Method
    ------------------------------------------
    function [x] = conjgrad(A, b, x)
        r = b - A * x;
        p = r;
        rsold = r' * r;

        for i = 1:length(b)
            Ap = A * p;
            alpha = rsold / (p' * Ap);
            x = x + alpha * p;
            r = r - alpha * Ap;
            rsnew = r' * r;
            if sqrt(rsnew) < 1e-10
                  break;
            end
            p = r + (rsnew / rsold) * p;
            rsold = rsnew;
        end
    end
    """

    def __init__(self, max_iterations, damping):
        assert damping >= 0.0
        self.damping = damping

        super(ConjugateGradient, self).__init__(max_iterations=max_iterations)

    def tf_solve(self, fn_x, x_init, b):
        return super(ConjugateGradient, self).tf_solve(fn_x, x_init, b)

    def tf_initialize(self, x_init, b):
        if x_init is None:
            x_init = [tf.zeros(shape=util.shape(t)) for t in b]
        conjugate = residual = [t - fx for t, fx in zip(b, self.fn_x(x=x_init))]
        square_residual = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(res * res)) for res in residual])
        initial_args = super(ConjugateGradient, self).tf_initialize(x_init)
        return initial_args + (conjugate, residual, square_residual)

    def tf_step(self, x, iteration, conjugate, residual, square_residual):
        #TODO Comments
        x, iteration, conjugate, residual, square_residual = super(ConjugateGradient, self).tf_step(
            x,
            iteration,
            conjugate,
            residual,
            square_residual
        )

        A_conjugate = self.fn_x(x=conjugate)

        if self.damping > 0.0:
            A_conjugate = [A_conj + self.damping * conj for A_conj, conj in zip(A_conjugate, conjugate)]

        conjugate_A_conjugate = tf.add_n(
            inputs=[tf.reduce_sum(input_tensor=(conj * A_conj)) for conj, A_conj in zip(conjugate, A_conjugate)]
        )

        alpha = square_residual / tf.maximum(x=conjugate_A_conjugate, y=util.epsilon)
        x = [t + alpha * conj for t, conj in zip(x, conjugate)]
        residual = [res - alpha * A_conj for res, A_conj in zip(residual, A_conjugate)]
        next_square_residual = tf.add_n(inputs=[tf.reduce_sum(input_tensor=(res * res)) for res in residual])

        conjugate = [res + (next_square_residual / tf.maximum(x=square_residual, y=util.epsilon)) * conj
                     for res, conj in zip(residual, conjugate)]
        square_residual = next_square_residual

        return x, iteration, conjugate, residual, square_residual

    def tf_next_step(self, x, iteration, conjugate, residual, square_residual):
        next_step = super(ConjugateGradient, self).tf_next_step(x, iteration, conjugate, residual, square_residual)
        return tf.logical_and(x=next_step, y=(square_residual >= util.epsilon))
