# Copyright 2017 reinforce.io. All Rights Reserved.
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