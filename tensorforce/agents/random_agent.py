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
Random agent that always returns a random action.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from random import gauss, random, randrange

from tensorforce.core import Agent


class RandomAgent(Agent):

    name = 'RandomAgent'
    model = (lambda config: None)

    def __init__(self, config):
        super(RandomAgent, self).__init__(config)

    def reset(self):
        self.episode += 1

    def act(self, state):
        """
        Get random action from action space

        :param state: current state (disregarded)
        :return: random action
        """
        self.timestep += 1
        actions = dict()
        for name, action in self.actions_config.items():
            if action.continuous:
                action = random()
                if 'min_value' in action:
                    action = action.min_value + random() * (action.max_value - action.min_value)
                else:
                    action = gauss(mu=0.0, sigma=1.0)
            else:
                action = randrange(action.num_actions)
            if self.unique_action:
                return action
            actions[name] = action
        return actions

    def observe(self, state, action, reward, terminal):
        pass
