# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Random agent that always returns a random action.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce.agents import RLAgent
from tensorforce.util.experiment_util import global_seed


class RandomAgent(RLAgent):
    name = 'RandomAgent'

    def __init__(self, config, scope):
        super(RandomAgent, self).__init__()

        if config.deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        self.actions = config.actions

    def get_action(self, state):
        """
        Get random action from action space

        :param state: current state (disregarded)
        :return: random action
        """
        return self.random.randint(0, self.actions)

    def add_observation(self, *args, **kwargs):
        pass

    def load_model(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass
