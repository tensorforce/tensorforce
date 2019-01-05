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

from tensorforce.agents.agent import Agent

from tensorforce.agents.constant_agent import ConstantAgent
from tensorforce.agents.drl_agent import DRLAgent
from tensorforce.agents.random_agent import RandomAgent

from tensorforce.agents.ddpg_agent import DDPGAgent
from tensorforce.agents.dqfd_agent import DQFDAgent
from tensorforce.agents.dqn_agent import DQNAgent
from tensorforce.agents.dqn_nstep_agent import DQNNstepAgent
from tensorforce.agents.naf_agent import NAFAgent
from tensorforce.agents.ppo_agent import PPOAgent
from tensorforce.agents.trpo_agent import TRPOAgent
from tensorforce.agents.vpg_agent import VPGAgent


agents = dict(
    constant=ConstantAgent, ddpg=DDPGAgent, dqfd=DQFDAgent, dqn=DQNAgent, dqn_nstep=DQNNstepAgent,
    naf=NAFAgent, ppo=PPOAgent, random=RandomAgent, trpo=TRPOAgent, vpg=VPGAgent
)


__all__ = [
    'Agent', 'agents', 'ConstantAgent', 'DDPGAgent', 'DQFDAgent', 'DQNAgent', 'DQNNstepAgent',
    'DRLAgent', 'NAFAgent', 'PPOAgent', 'RandomAgent', 'TRPOAgent', 'VPGAgent'
]
