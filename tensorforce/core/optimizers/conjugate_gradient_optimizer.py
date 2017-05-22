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
"""
Simple conjugate gradient solver, as used in the TRPO implementations of Schulman
and others. This code is under MIT license, for more information see LICENSE-EXT.

Conjugate gradients solve linear systems of equations Ax=b through iteratively constructing n conjugate
search directions, thus guaranteeing convergence in at most n steps, although in practice much fewer steps are
used for large sparse systems.

The key idea of cg ist that the next conjugate vector p_k can be computed
just based on the previous search direction as a linear combination of the negative residual and previous
search direction, instead of using a memory intensive orthogonalization process such as Gram-Schmidt.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np

from tensorforce import util


# TODO This should ultimately be refactored to do a full constrainted optimization as in rllab
class ConjugateGradientOptimizer(object):

    def __init__(self, logger=None, cg_iterations=10, stop_residual=1e-10):
        self.logger = logger
        self.iterations = cg_iterations
        self.stop_residual = stop_residual

    def solve(self, f_Ax, b):
        """Conjugate gradient solver.

        
        Args:
            f_Ax: Ax of Ax=b
            b: b in Ax = b

        Returns: Approximate solution to linear system.

        """
        b = np.nan_to_num(b)
        cg_vector_p = b.copy()
        residual = b.copy()
        x = np.zeros_like(b)
        residual_dot_residual = residual.dot(residual)

        for i in xrange(self.iterations):
            z = f_Ax(cg_vector_p)
            cg_vector_p_dot_z = cg_vector_p.dot(z)
            if abs(cg_vector_p_dot_z) < util.epsilon:
                cg_vector_p_dot_z = util.epsilon
            v = residual_dot_residual / cg_vector_p_dot_z
            x += v * cg_vector_p

            residual -= v * z
            new_residual_dot_residual = residual.dot(residual)
            alpha = new_residual_dot_residual / (residual_dot_residual + util.epsilon)

            # Construct new search direction as linear combination of residual and previous
            # search vector.
            cg_vector_p = residual + alpha * cg_vector_p
            residual_dot_residual = new_residual_dot_residual

            if residual_dot_residual < self.stop_residual:
                self.logger.debug('Approximate cg solution found after {:d} iterations'.format(i + 1))
                break

        return np.nan_to_num(x)
