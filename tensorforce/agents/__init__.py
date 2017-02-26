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

from tensorforce.agents.rl_agent import RLAgent
from tensorforce.agents.random_agent import RandomAgent
from tensorforce.agents.memory_agent import MemoryAgent
from tensorforce.agents.dqn_agent import DQNAgent
from tensorforce.agents.naf_agent import NAFAgent
from tensorforce.agents.pg_agent import PGAgent
from tensorforce.agents.trpo_agent import TRPOAgent
from tensorforce.agents.vpg_agent import VPGAgent

__all__ = ['RLAgent', 'RandomAgent', 'MemoryAgent', 'DQNAgent', 'NAFAgent',
           'PGAgent', 'TRPOAgent', 'VPGAgent']
