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
OpenAI Gym Integration: https://gym.openai.com/.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
from gym.wrappers import Monitor
from gym.spaces.discrete import Discrete
from tensorforce.environments import Environment


class OpenAIGymEnvironment(Environment):

    def __init__(self, gym_id, monitor=None, monitor_safe=False, monitor_video=0):
        """
        Initialize open ai gym environment.

        :param gym_id: string with id/descriptor of the gym environment, e.g. 'CartPole-v0'
        :param monitor:
        """
        self.gym_id = gym_id
        self.gym = gym.make(gym_id)  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv

        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = lambda x: x % monitor_video == 0
            self.gym = Monitor(self.gym, monitor, force=not monitor_safe, video_callable=video_callable)

    def __str__(self):
        return self.gym_id

    def monitor(self, path):
        self.gym = Monitor(self.gym, path)

    def reset(self):
        """
        Pass reset function to gym.

        :return: ndarray containing initial state
        """
        return self.gym.reset()

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.gym = None

    def execute_action(self, action):
        """
        Pass action to gym, return reward, next step, terminal state and additional info.

        :param action: Action to execute
        :return: dict containing the next state, the reward, and a boolean indicating
            if the next state is a terminal state, as well as additional information provided by the gym
        """
        state, reward, terminal_state, info = self.gym.step(action)

        return dict(state=state,
                    reward=reward,
                    terminal_state=terminal_state,
                    info=info)

    @property
    def actions(self):
        if isinstance(self.gym.action_space, Discrete):
            return self.gym.action_space.n
        else:
            return self.gym.action_space.shape[0]

    @property
    def action_shape(self):
        if isinstance(self.gym.action_space, Discrete):
            return []
        else:
            return (self.gym.action_space.shape[0],)

    @property
    def state_shape(self):
        return self.gym.observation_space.shape
