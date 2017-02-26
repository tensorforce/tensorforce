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
Standard DQN. The piece de resistance of deep reinforcement learning.
Chooses from one of a number of discrete actions by taking the maximum Q-value
from the value function with one output neuron per available action.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import MemoryAgent
from tensorforce.models import DQNModel

from tensorforce.default_configs import DQNAgentConfig


class DQNAgent(MemoryAgent):
    name = 'DQNAgent'

    default_config = DQNAgentConfig

    model_ref = DQNModel
