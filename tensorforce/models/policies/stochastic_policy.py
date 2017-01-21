"""
Generic stochastic policy for policy gradients.
"""
from tensorforce.models.policies.categorical import Categorical
from tensorforce.models.policies.gaussian_policy import GaussianPolicy


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
        self.session = session
        self.neural_network = neural_network
        self.state = state
        self.action_count = action_count
        self.random = random

    def sample(self, state):
        raise NotImplementedError

    def get_distribution(self):
        raise NotImplementedError

    def get_output_variables(self):
        raise NotImplementedError


stochastic_policies = {
    'gaussian' : GaussianPolicy,
    'categorical': Categorical
}