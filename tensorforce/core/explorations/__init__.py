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


from tensorforce.core.explorations.exploration import Exploration
from tensorforce.core.explorations.constant import Constant
from tensorforce.core.explorations.linear_decay import LinearDecay
from tensorforce.core.explorations.epsilon_anneal import EpsilonAnneal
from tensorforce.core.explorations.epsilon_decay import EpsilonDecay
from tensorforce.core.explorations.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess


explorations = dict(
    constant=Constant,
    linear_decay=LinearDecay,
    epsilon_anneal=EpsilonAnneal,
    epsilon_decay=EpsilonDecay,
    ornstein_uhlenbeck=OrnsteinUhlenbeckProcess
)


__all__ = ['Exploration', 'Constant', 'LinearDecay', 'EpsilonDecay', 'OrnsteinUhlenbeckProcess', 'explorations']
