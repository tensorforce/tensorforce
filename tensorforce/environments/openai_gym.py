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

import numpy as np

from tensorforce import TensorForceError
from tensorforce.environments import Environment


class OpenAIGym(Environment):

    def __init__(self, gym_id, monitor=None, monitor_safe=False, monitor_video=0):
        """
        Initialize OpenAI gym environment.

        :param gym_id: OpenAI Gym environment ID. See https://gym.openai.com/envs
        :param monitor: Output directory. Setting this to None disables monitoring.
        :param monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
        :param monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.
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
        return 'OpenAIGym({})'.format(self.gym_id)

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.gym = None

    def reset(self):
        """
        Pass reset function to gym.
        """
        return self.gym.reset()

    def execute(self, action):
        """
        Pass action to gym, return reward, next step, terminal state and additional info.
        """
        if isinstance(self.gym.action_space, gym.spaces.Box):
            action = [action] # some gym environments expect a list (f.i. Pendulum-v0)
        state, reward, terminal, _ = self.gym.step(action)
        return state, reward, terminal

    @property
    def states(self):
        if isinstance(self.gym.observation_space, Discrete):
            return dict(shape=(), type='float')
        else:
            return dict(shape=tuple(self.gym.observation_space.shape), type='float')

    @property
    def actions(self):
        if isinstance(self.gym.action_space, Discrete):
            return dict(continuous=False, num_actions=self.gym.action_space.n)
        elif len(self.gym.action_space.shape) == 1:
            return dict(continuous=True)
        elif len(self.gym.action_space.shape) > 1:
            return {'action' + str(n): dict(continuous=True) for n in range(len(self.gym.action_space.shape))}
        else:
            raise TensorForceError()

    def monitor(self, path):
        self.gym = Monitor(self.gym, path)
