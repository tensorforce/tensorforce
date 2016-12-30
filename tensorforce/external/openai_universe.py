# Copyright 2016 reinforce.io. All Rights Reserved.
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
OpenAI Universe Integration: https://universe.openai.com/.
Contains OpenAI Gym: https://gym.openai.com/.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import universe
from gym.spaces.discrete import Discrete
from tensorforce.environments.environment import Environment


class OpenAIUniverseEnvironment(Environment):
    def __init__(self, env_id):
        """
        Initialize open ai universe environment.

        :param env_id: string with id/descriptor of the universe environment, e.g. 'HarvestDay-v0'
        """
        self.env_id = env_id
        self.env = gym.make(env_id)

    def reset(self):
        """
        Pass reset function to universe environment.

        :return: ndarray containing initial state
        """
        return self.env.reset()

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.env = None

    def execute_action(self, action):
        """
        Pass action to universe environment, return reward, next step, terminal state and additional info.

        :param action: Action to execute
        :return: dict containing next_state, reward, and a boolean indicating
            if next state is a terminal state, as well as additional information provided by the universe environment
        """
        state, reward, terminal_state, info = self.env.step(action)

        return dict(state=state,
                    reward=reward,
                    terminal_state=terminal_state,
                    info=info)

    @property
    def actions(self):
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    @property
    def action_shape(self):
        if isinstance(self.env.action_space, Discrete):
            return []
        else:
            return (self.env.action_space.shape[0],)

    @property
    def state_shape(self):
        return self.env.observation_space.shape
