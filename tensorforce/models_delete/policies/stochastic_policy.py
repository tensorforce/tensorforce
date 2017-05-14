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


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class StochasticPolicy(object):
    """
    Generic stochastic policy for policy gradients.
    """

    def __init__(self, network, policy_outputs, session, state, random, action_count=1):
        """
        Stochastic policy for sampling and updating utilities.

        :param neural_network: Handle to policy network used for prediction
        """
        self.policy_outputs = policy_outputs
        self.episode_length = network.episode_length
        self.internal_state_inputs = network.internal_state_inputs
        self.internal_state_outputs = network.internal_state_outputs
        self.internal_states = network.internal_state_inits
        self.session = session
        self.state = state
        self.action_count = action_count
        self.random = random

    def sample(self, states):
        fetches = list(self.policy_outputs)
        fetches.extend(self.internal_state_outputs)

        feed_dict = {self.episode_length: [1], self.state: [(states,)]}
        feed_dict.update({internal_state: self.internal_states[n] for n,
                          internal_state in enumerate(self.internal_state_inputs)})

        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)
        sample = fetched[:len(self.policy_outputs)]

        self.internal_states = fetched[len(self.policy_outputs):]

        return sample

    def get_distribution(self):
        raise NotImplementedError

    def get_policy_variables(self):
        raise NotImplementedError
