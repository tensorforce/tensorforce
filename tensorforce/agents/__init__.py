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

from tensorforce.agents.constant import ConstantAgent
from tensorforce.agents.random import RandomAgent
from tensorforce.agents.tensorforce import TensorforceAgent

from tensorforce.agents.a2c import AdvantageActorCritic
from tensorforce.agents.ac import ActorCritic
from tensorforce.agents.dpg import DeterministicPolicyGradient
from tensorforce.agents.dqn import DeepQNetwork
from tensorforce.agents.dueling_dqn import DuelingDQN
from tensorforce.agents.ppo import ProximalPolicyOptimization
from tensorforce.agents.trpo import TrustRegionPolicyOptimization
from tensorforce.agents.vpg import VanillaPolicyGradient


A2C = A2CAgent = AdvantageActorCritic
AC = ACAgent = ActorCritic
DPG = DPGAgent = DeterministicPolicyGradient
DQN = DQNAgent = DeepQNetwork
DuelingDQNAgent = DuelingDQN
PPO = PPOAgent = ProximalPolicyOptimization
TRPO = TRPOAgent = TrustRegionPolicyOptimization
VPG = VPGAgent = REINFORCE = VanillaPolicyGradient


agents = dict(
    a2c=AdvantageActorCritic, ac=ActorCritic, constant=ConstantAgent,
    dpg=DeterministicPolicyGradient, dqn=DeepQNetwork, dueling_dqn=DuelingDQN,
    tensorforce=TensorforceAgent, ppo=ProximalPolicyOptimization, random=RandomAgent,
    reinforce=VanillaPolicyGradient, trpo=TrustRegionPolicyOptimization, vpg=VanillaPolicyGradient
)


__all__ = [
    'A2C', 'A2CAgent', 'AC', 'ACAgent', 'ActorCritic', 'AdvantageActorCritic', 'Agent', 'agents',
    'ConstantAgent', 'DeepQNetwork', 'DeterministicPolicyGradient', 'DPG', 'DPGAgent', 'DQN',
    'DQNAgent', 'DuelingDQN', 'DuelingDQNAgent', 'PPO', 'PPOAgent', 'ProximalPolicyOptimization',
    'RandomAgent', 'REINFORCE', 'TensorforceAgent', 'TRPO', 'TRPOAgent',
    'TrustRegionPolicyOptimization', 'VanillaPolicyGradient', 'VPG', 'VPGAgent'
]
