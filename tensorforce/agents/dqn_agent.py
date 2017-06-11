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
Standard DQN. The piece de resistance of deep reinforcement learning.
Chooses from one of a number of discrete actions by taking the maximum Q-value
from the value function with one output neuron per available action.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.core import MemoryAgent
from tensorforce.models import DQNModel


class DQNAgent(MemoryAgent):

    name = 'DQNAgent'
    model = DQNModel
    default_config = dict(
        target_update_frequency=10000
    )

    def __init__(self, config):
        config.default(MemoryAgent.default_config)
        super(DQNAgent, self).__init__(config)
        self.target_update_frequency = config.target_update_frequency

    def observe(self, state, action, reward, terminal):
        super(DQNAgent, self).observe(state=state, action=action, reward=reward, terminal=terminal)
        if self.timestep >= self.first_update and self.timestep % self.target_update_frequency == 0:
            self.model.update_target()
