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
import numpy as np
from tensorforce import TensorForceError
from tensorforce.environments import Environment


class OpenAIGym(Environment):

    def __init__(self, gym_id, monitor=None, monitor_safe=False, monitor_video=0):
        """
        Initialize OpenAI Gym.

        Args:
            gym_id: OpenAI Gym environment ID. See https://gym.openai.com/envs
            monitor: Output directory. Setting this to None disables monitoring.
            monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.
        """

        self.gym_id = gym_id
        self.gym = gym.make(gym_id)  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv

        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            self.gym = gym.wrappers.Monitor(self.gym, monitor, force=not monitor_safe, video_callable=video_callable)

    def __str__(self):
        return 'OpenAIGym({})'.format(self.gym_id)

    def close(self):
        self.gym.close()
        self.gym = None

    def reset(self):
        if isinstance(self.gym, gym.wrappers.Monitor):
            self.gym.stats_recorder.done = True
        return self.gym.reset()

    def execute(self, actions):
        state, reward, terminal, _ = self.gym.step(actions)
        return state, terminal, reward

    @property
    def states(self):
        return OpenAIGym.state_from_space(space=self.gym.observation_space)

    @staticmethod
    def state_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=(), type='int')
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(shape=space.n, type='int')
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.num_discrete_space, type='int')
        elif isinstance(space, gym.spaces.Box):
            return dict(shape=tuple(space.shape), type='float')
        elif isinstance(space, gym.spaces.Tuple):
            states = dict()
            n = 0
            for space in space.spaces:
                state = OpenAIGym.state_from_space(space=space)
                if 'type' in state:
                    states['state{}'.format(n)] = state
                    n += 1
                else:
                    for state in state.values():
                        states['state{}'.format(n)] = state
                        n += 1
            return states
        else:
            raise TensorForceError('Unknown Gym space.')

    @property
    def actions(self):
        return OpenAIGym.action_from_space(space=self.gym.action_space)

    @staticmethod
    def action_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', num_actions=space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(type='int', num_actions=(space.high[0] - space.low[0]), shape=space.num_discrete_space)
            else:
                actions = dict()
                for n in range(space.num_discrete_space):
                    actions['action{}'.format(n)] = dict(type='int', num_actions=(space.high[n] - space.low[n]))
                return actions
        elif isinstance(space, gym.spaces.Box):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(type='float', shape=space.low.shape,
                            min_value=np.float32(space.low[0]),
                            max_value=np.float32(space.high[0]))
            else:
                actions = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    actions['action{}'.format(n)] = dict(type='float', min_value=low[n], max_value=high[n])
                return actions
        elif isinstance(space, gym.spaces.Tuple):
            actions = dict()
            n = 0
            for space in space.spaces:
                action = OpenAIGym.action_from_space(space=space)
                if 'type' in action:
                    actions['action{}'.format(n)] = action
                    n += 1
                else:
                    for action in action.values():
                        actions['action{}'.format(n)] = action
                        n += 1
            return actions
        else:
            raise TensorForceError('Unknown Gym space.')
