import numpy as np

"""
Basic Reinforcement learning agent. An agent encapsulates execution logic
of a particular reinforcement learning algorithm and defines the external interface
to the environment. The agent hence acts an intermediate layer between environment
and backend execution (value function or policy updates).
"""


class RLAgent(object):
    pass

    def execute_step(self, state, reward, is_terminal):
        raise NotImplementedError

