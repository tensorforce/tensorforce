# Copyright 2016 reinforce.io. All Rights Reserved.
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
"""
Simple conjugate gradient solver, as used in the TRPO implementations of Schulman
and others.

Conjugate gradients solve linear systems of equations Ax=b through iteratively constructing n conjugate
search directions, thus guaranteeing convergence in at most n steps, although in practice much fewer steps are
used for large sparse systems. The key idea of cg ist that the next conjugate vector p_k can be computed
just based on the previous search direction as a linear combination of the negative residual and previous
search direction, instead of using a memory intensive orthogonalization process such as Gram-Schmidt.
"""
from six.moves import xrange
import numpy as np


# TODO This should ultimately be refactored to do a full constrainted optimization as in rllab


class ConjugateGradientOptimizer(object):
    def __init__(self, cg_iterations=10, stop_residual=1e-10):
        self.iterations = cg_iterations
        self.stop_residual = stop_residual

    def solve(self, f_Ax, b):
        """
        Conjugate gradient solver.

        :param f_Ax: Ax of Ax=b
        :param b: b in Ax = b
        :return:
        """

        cg_vector_p = b.copy()
        residual = b.copy()
        x = np.zeros_like(b)
        residual_dot_residual = residual.dot(residual)

        for i in xrange(self.iterations):
            z = f_Ax(cg_vector_p)
            v = residual_dot_residual / cg_vector_p.dot(z)
            x += v * cg_vector_p

            residual -= v * z
            new_residual_dot_residual = residual.dot(residual)
            alpha = new_residual_dot_residual / residual_dot_residual

            cg_vector_p = residual + alpha * cg_vector_p
            residual_dot_residual = new_residual_dot_residual

            if residual_dot_residual < self.stop_residual:
                print('Approximate cg solution found after ' + str(i) + ' iterations')
                break

        print('')
        return x
