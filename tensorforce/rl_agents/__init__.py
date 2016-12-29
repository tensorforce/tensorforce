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

from tensorforce.rl_agents.rl_agent import RLAgent
from tensorforce.rl_agents.random_agent import RandomAgent
from tensorforce.rl_agents.memory_agent import MemoryAgent
from tensorforce.rl_agents.dqn_agent import DQNAgent
from tensorforce.rl_agents.naf_agent import NAFAgent
from tensorforce.rl_agents.pg_agent import PGAgent
from tensorforce.rl_agents.trpo_agent import TRPOAgent

__all__ = ['RLAgent', 'RandomAgent', 'MemoryAgent', 'DQNAgent', 'NAFAgent', 'PGAgent', 'TRPOAgent']
