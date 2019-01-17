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

from tensorforce.core.parameters.parameter import Parameter

from tensorforce.core.parameters.constant import Constant
from tensorforce.core.parameters.decaying import Decaying
from tensorforce.core.parameters.ornstein_uhlenbeck import OrnsteinUhlenbeck
from tensorforce.core.parameters.piecewise_constant import PiecewiseConstant
from tensorforce.core.parameters.random import Random


parameter_modules = dict(
    constant=Constant, decaying=Decaying, default=Constant, ornstein_uhlenbeck=OrnsteinUhlenbeck,
    piecewise_constant=PiecewiseConstant, random=Random
)


__all__ = [
    'Constant', 'Decaying', 'OrnsteinUhlenbeck', 'Parameter', 'parameter_modules',
    'PiecewiseConstant', 'Random'
]
