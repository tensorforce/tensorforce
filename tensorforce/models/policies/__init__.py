# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

from tensorforce.models.policies.distribution import Distribution
from tensorforce.models.policies.categorical import Categorical
from tensorforce.models.policies.gaussian import Gaussian
from tensorforce.models.policies.stochastic_policy import StochasticPolicy
from tensorforce.models.policies.categorical_one_hot_policy import CategoricalOneHotPolicy
from tensorforce.models.policies.gaussian_policy import GaussianPolicy


__all__ = [ 'Distribution','Categorical','Gaussian','StochasticPolicy', 'CategoricalOneHot',
           'GaussianPolicy']

stochastic_policies = {
    'gaussian': GaussianPolicy,
    'categorical': Categorical
}