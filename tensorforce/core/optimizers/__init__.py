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

from functools import partial
from tensorforce.core.optimizers.optimizer import Optimizer
from tensorforce.core.optimizers.meta_optimizer import MetaOptimizer
from tensorforce.core.optimizers.global_optimizer import GlobalOptimizer
from tensorforce.core.optimizers.tf_optimizer import TFOptimizer
from tensorforce.core.optimizers.evolutionary import Evolutionary
from tensorforce.core.optimizers.natural_gradient import NaturalGradient
from tensorforce.core.optimizers.kfac import KFAC
from tensorforce.core.optimizers.clipped_step import ClippedStep
from tensorforce.core.optimizers.multi_step import MultiStep
from tensorforce.core.optimizers.optimized_step import OptimizedStep
from tensorforce.core.optimizers.subsampling_step import SubsamplingStep
from tensorforce.core.optimizers.synchronization import Synchronization


# This can register any class inheriting from tf.train.Optimizer
optimizers = dict(
    global_optimizer=GlobalOptimizer,
    adadelta=partial(TFOptimizer, 'adadelta'),
    adagrad=partial(TFOptimizer, 'adagrad'),
    adam=partial(TFOptimizer, 'adam'),
    nadam=partial(TFOptimizer, 'nadam'),
    gradient_descent=partial(TFOptimizer, 'gradient_descent'),
    momentum=partial(TFOptimizer, 'momentum'),
    rmsprop=partial(TFOptimizer, 'rmsprop'),
    evolutionary=Evolutionary,
    natural_gradient=NaturalGradient,
    kfac=KFAC,
    clipped_step=ClippedStep,
    multi_step=MultiStep,
    optimized_step=OptimizedStep,
    subsampling_step=SubsamplingStep,
    synchronization=Synchronization
)


__all__ = [
    'optimizers',
    'Optimizer',
    'MetaOptimizer',
    'GlobalOptimizer',
    'TFOptimizer',
    'Evolutionary',
    'NaturalGradient',
    'ClippedStep',
    'MultiStep',
    'OptimizedStep',
    'SubsamplingStep',
    'Synchronization'
]
