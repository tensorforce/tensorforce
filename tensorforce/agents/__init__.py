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

from tensorforce.agents.agent import Agent
from tensorforce.agents.batch_agent import BatchAgent
from tensorforce.agents.constant_agent import ConstantAgent
from tensorforce.agents.memory_agent import MemoryAgent
from tensorforce.agents.random_agent import RandomAgent
from tensorforce.agents.vpg_agent import VPGAgent
from tensorforce.agents.trpo_agent import TRPOAgent
from tensorforce.agents.ppo_agent import PPOAgent
from tensorforce.agents.dqn_agent import DQNAgent
from tensorforce.agents.ddqn_agent import DDQNAgent
from tensorforce.agents.dqn_nstep_agent import DQNNstepAgent
from tensorforce.agents.naf_agent import NAFAgent
from tensorforce.agents.dqfd_agent import DQFDAgent
# from tensorforce.agents.categorical_dqn_agent import CategoricalDQNAgent

agents = dict(
    constant_agent=ConstantAgent,
    random_agent=RandomAgent,
    vpg_agent=VPGAgent,
    trpo_agent=TRPOAgent,
    ppo_agent=PPOAgent,
    dqn_agent=DQNAgent,
    ddqn_agent=DDQNAgent,
    dqn_nstep_agent=DQNNstepAgent,
    naf_agent=NAFAgent,
    dqfd_agent=DQFDAgent
    # CategoricalDQNAgent=CategoricalDQNAgent,
)

__all__ = [
    'Agent',
    'BatchAgent',
    'MemoryAgent',
    'ConstantAgent',
    'RandomAgent',
    'VPGAgent',
    'TRPOAgent',
    'PPOAgent',
    'DQNAgent',
    'DDQNAgent',
    'DQNNstepAgent',
    'DQFDAgent',
    'NAFAgent',
    'agents'
]
