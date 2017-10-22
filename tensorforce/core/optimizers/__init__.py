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

from tensorforce.core.optimizers.optimizer import Optimizer
from tensorforce.core.optimizers.meta_optimizer import MetaOptimizer
from tensorforce.core.optimizers.tf_optimizer import TFOptimizer
from tensorforce.core.optimizers.evolutionary import Evolutionary
from tensorforce.core.optimizers.natural_gradient import NaturalGradient
from tensorforce.core.optimizers.multi_step import MultiStep
from tensorforce.core.optimizers.optimized_step import OptimizedStep
from tensorforce.core.optimizers.synchronization import Synchronization
from tensorforce.core.optimizers.global_optimizer import GlobalOptimizer


# This can register any class inheriting from tf.train.Optimizer
optimizers = dict(
    adadelta=TFOptimizer.get_wrapper(optimizer='adadelta'),
    adagrad=TFOptimizer.get_wrapper(optimizer='adagrad'),
    adam=TFOptimizer.get_wrapper(optimizer='adam'),
    nadam=TFOptimizer.get_wrapper(optimizer='nadam'),
    gradient_descent=TFOptimizer.get_wrapper(optimizer='gradient_descent'),
    momentum=TFOptimizer.get_wrapper(optimizer='momentum'),
    rmsprop=TFOptimizer.get_wrapper(optimizer='rmsprop'),
    evolutionary=Evolutionary,
    natural_gradient=NaturalGradient,
    multi_step=MultiStep,
    optimized_step=OptimizedStep,
    synchronization=Synchronization
    # GlobalOptimizer not (yet) a valid choice
)


__all__ = ['optimizers', 'Optimizer', 'MetaOptimizer', 'TFOptimizer', 'Evolutionary', 'NaturalGradient', 'MultiStep', 'OptimizedStep', 'Synchronization', 'GlobalOptimizer']
