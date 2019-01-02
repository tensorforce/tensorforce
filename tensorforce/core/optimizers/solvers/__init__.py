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

from tensorforce.core.optimizers.solvers.solver import Solver
from tensorforce.core.optimizers.solvers.iterative import Iterative

from tensorforce.core.optimizers.solvers.conjugate_gradient import ConjugateGradient
from tensorforce.core.optimizers.solvers.line_search import LineSearch


solver_modules = dict(conjugate_gradient=ConjugateGradient, line_search=LineSearch)


__all__ = ['ConjugateGradient', 'Iterative', 'LineSearch', 'Solver', 'solver_modules']
