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

from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorforce.core.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


optimizers = dict(
    adadelta=AdadeltaOptimizer,
    adagrad=AdagradOptimizer,
    adam=AdamOptimizer,
    gradient_descent=GradientDescentOptimizer,
    momentum=MomentumOptimizer,
    rmsprop=RMSPropOptimizer,
    conjugate_gradient=ConjugateGradientOptimizer
)


__all__ = ['optimizers', 'AdadeltaOptimizer', 'AdagradOptimizer', 'AdamOptimizer', 'GradientDescentOptimizer', 'MomentumOptimizer', 'RMSPropOptimizer', 'ConjugateGradientOptimizer']
