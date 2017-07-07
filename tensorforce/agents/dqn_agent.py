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
Standard DQN agent.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.agents import MemoryAgent
from tensorforce.models import DQNModel


class DQNAgent(MemoryAgent):
    """
    Deep-Q-Network agent (DQN). The piece de resistance of deep reinforcement learning as described by
    [Minh et al. (2015)](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)). Includes
    an option for double-DQN (DDQN; [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)))

    DQN chooses from one of a number of discrete actions by taking the maximum Q-value
    from the value function with one output neuron per available action. DQN uses a replay memory for experience
    playback.

    """

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
