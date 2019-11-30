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

from functools import partial

from tensorforce.core.optimizers.optimizer import Optimizer

from tensorforce.core.optimizers.meta_optimizer import MetaOptimizer

from tensorforce.core.optimizers.clipping_step import ClippingStep
from tensorforce.core.optimizers.evolutionary import Evolutionary
from tensorforce.core.optimizers.global_optimizer import GlobalOptimizer
from tensorforce.core.optimizers.meta_optimizer_wrapper import MetaOptimizerWrapper
from tensorforce.core.optimizers.multi_step import MultiStep
from tensorforce.core.optimizers.natural_gradient import NaturalGradient
from tensorforce.core.optimizers.optimizing_step import OptimizingStep
from tensorforce.core.optimizers.plus import Plus
from tensorforce.core.optimizers.subsampling_step import SubsamplingStep
from tensorforce.core.optimizers.synchronization import Synchronization
from tensorforce.core.optimizers.tf_optimizer import TFOptimizer, tensorflow_optimizers


optimizer_modules = dict(
    clipping_step=ClippingStep, default=MetaOptimizerWrapper, evolutionary=Evolutionary,
    global_optimizer=GlobalOptimizer, meta_optimizer_wrapper=MetaOptimizerWrapper,
    multi_step=MultiStep, natural_gradient=NaturalGradient, optimizing_step=OptimizingStep,
    plus=Plus, subsampling_step=SubsamplingStep, synchronization=Synchronization,
    tf_optimizer=TFOptimizer
)


for name, optimizer in tensorflow_optimizers.items():
    assert name not in optimizer_modules
    optimizer_modules[name] = partial(TFOptimizer, optimizer=name)


__all__ = [
    'ClippingStep', 'Evolutionary', 'GlobalOptimizer', 'MetaOptimizer', 'MetaOptimizerWrapper',
    'MultiStep', 'NaturalGradient', 'OptimizingStep', 'Optimizer', 'optimizer_modules', 'Plus',
    'SubsamplingStep', 'Synchronization', 'TFOptimizer'
]
