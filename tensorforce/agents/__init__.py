# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce.agents.recorder import Recorder

from tensorforce.agents.agent import Agent

from tensorforce.agents.constant import ConstantAgent
from tensorforce.agents.random import RandomAgent
from tensorforce.agents.tensorforce import TensorforceAgent

from tensorforce.agents.a2c import AdvantageActorCritic
from tensorforce.agents.ac import ActorCritic
from tensorforce.agents.dpg import DeterministicPolicyGradient
from tensorforce.agents.double_dqn import DoubleDQN
from tensorforce.agents.dqn import DeepQNetwork
from tensorforce.agents.dueling_dqn import DuelingDQN
from tensorforce.agents.ppo import ProximalPolicyOptimization
from tensorforce.agents.trpo import TrustRegionPolicyOptimization
from tensorforce.agents.vpg import VanillaPolicyGradient


A2C = A2CAgent = AdvantageActorCritic
AC = ACAgent = ActorCritic
Constant = ConstantAgent
DPG = DDPG = DPGAgent = DeterministicPolicyGradient
DDQN = DoubleDQNAgent = DoubleDQN
DQN = DQNAgent = DeepQNetwork
DuelingDQNAgent = DuelingDQN
PPO = PPOAgent = ProximalPolicyOptimization
Random = RandomAgent
Tensorforce = TensorforceAgent
TRPO = TRPOAgent = TrustRegionPolicyOptimization
VPG = REINFORCE = VPGAgent = VanillaPolicyGradient


agents = dict(
    a2c=AdvantageActorCritic, ac=ActorCritic, constant=ConstantAgent,
    ddpg=DeterministicPolicyGradient, ddqn=DoubleDQN, default=TensorforceAgent,
    dpg=DeterministicPolicyGradient, double_dqn=DoubleDQN, dqn=DeepQNetwork, dueling_dqn=DuelingDQN,
    tensorforce=TensorforceAgent, ppo=ProximalPolicyOptimization, random=RandomAgent,
    recorder=Recorder, reinforce=VanillaPolicyGradient, trpo=TrustRegionPolicyOptimization,
    vpg=VanillaPolicyGradient
)


__all__ = [
    'Agent', 'agents',
    'A2C', 'A2CAgent', 'AdvantageActorCritic',
    'AC', 'ACAgent', 'ActorCritic',
    'Constant', 'ConstantAgent',
    'DPG', 'DDPG', 'DPGAgent', 'DeterministicPolicyGradient',
    'DDQN', 'DoubleDQNAgent', 'DoubleDQN',
    'DQN', 'DQNAgent', 'DeepQNetwork',
    'DuelingDQN', 'DuelingDQNAgent',
    'PPO', 'PPOAgent', 'ProximalPolicyOptimization',
    'Random', 'RandomAgent',
    'Tensorforce', 'TensorforceAgent',
    'TRPO', 'TRPOAgent', 'TrustRegionPolicyOptimization',
    'VPG', 'REINFORCE', 'VPGAgent', 'VanillaPolicyGradient'
]
