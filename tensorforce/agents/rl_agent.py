# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Basic Reinforcement learning agent. An agent encapsulates execution logic
of a particular reinforcement learning algorithm and defines the external interface
to the environment.

 The agent hence acts an intermediate layer between environment
and backend execution (value function or policy updates).
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class RLAgent(object):
    name = 'RLAgent'

    def get_action(self, state):
        raise NotImplementedError

    def add_observation(self, state, action, reward, terminal):
        raise NotImplementedError

    def get_variables(self):
        raise NotImplementedError

    def assign_variables(self, values):
        raise NotImplementedError

    def get_gradients(self):
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def __str__(self):
        return self.name
