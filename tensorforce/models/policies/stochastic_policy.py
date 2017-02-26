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

