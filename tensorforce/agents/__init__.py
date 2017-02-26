# Copyright 2016 reinforce.io. All Rights Reserved.
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
