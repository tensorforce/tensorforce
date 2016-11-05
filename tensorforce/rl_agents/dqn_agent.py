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

"""
Standard DQN. The piece de resistance of deep reinforcement learning.
Chooses from one of a number of discrete actions by taking the maximum Q-value
from the value function with one output neuron per available action.
"""

from tensorforce.replay_memories.replay_memory import ReplayMemory
from tensorforce.rl_agents.memory_agent import MemoryAgent
from tensorforce.rl_agents.rl_agent import RLAgent
from tensorforce.value_functions.deep_q_network import DeepQNetwork


class DQNAgent(MemoryAgent):

    def __init__(self, agent_config, network_config):
        """
        Initialize a vanilla DQN agent as described in
        http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html.

        :param agent_config: Configuration parameters for agent
        :param network_config: Configuration parameters for deep Q network,
        i.e. network configuration
        """
        super(DQNAgent, self).__init__(agent_config)
        self.value_function = DeepQNetwork(agent_config, network_config, agent_config['deterministic_mode'])

