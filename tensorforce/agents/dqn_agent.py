# Copyright 2016 reinforce.io. All Rights Reserved.
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
