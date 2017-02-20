# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================
"""
Generic stochastic policy for policy gradients.
"""

class StochasticPolicy(object):
    def __init__(self,
                 neural_network=None,
                 session=None,
                 state=None,
                 random=None,
                 action_count=1):
        """
        Stochastic policy for sampling and updating utilities.

        :param neural_network: Handle to policy network used for prediction
        """
        self.neural_network = neural_network
        self.session = session
        self.state = state
        self.action_count = action_count
        self.random = random

    def sample(self, state):
        raise NotImplementedError

    def get_distribution(self):
        raise NotImplementedError

    def get_policy_variables(self):
        raise NotImplementedError

