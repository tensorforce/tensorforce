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

from tensorforce.updater.model import Model
from tensorforce.updater.deep_q_network import DeepQNetwork
from tensorforce.updater.linear_value_function import LinearValueFunction
from tensorforce.updater.naf_network import NAFNetwork
from tensorforce.updater.trpo_updater import TRPOUpdater

__all__ = ['Model', 'DeepQNetwork', 'LinearValueFunction', 'NAFNetwork', 'TRPOUpdater']
