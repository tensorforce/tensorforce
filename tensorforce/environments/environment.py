# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Base environment class
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class Environment(object):

    def __str__(self):
        return 'Environment'

    def reset(self):
        """
        Reset environment and setup for new episode.

        :return: initial state
        """
        raise NotImplementedError

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        raise NotImplementedError

    def execute_action(self, action):
        """
        Executes action, observes next state and reward.

        :param action: Action to execute

        :return: dict containing at least next_state, reward, and terminal_state
        """
        raise NotImplementedError
