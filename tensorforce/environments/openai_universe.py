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
OpenAI Universe Integration: https://universe.openai.com/.
Contains OpenAI Gym: https://gym.openai.com/.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import universe
from gym.spaces.discrete import Discrete
from universe.spaces import VNCActionSpace, VNCObservationSpace

from tensorforce import TensorForceError
from tensorforce.environments.environment import Environment


class OpenAIUniverse(Environment):
    def __init__(self, env_id):
        """
        Initialize open ai universe environment.

        :param env_id: string with id/descriptor of the universe environment, e.g. 'HarvestDay-v0'
        """
        self.env_id = env_id
        self.env = gym.make(env_id)

    def __str__(self):
        return 'OpenAI-Universe({})'.format(self.env_id)

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.env = None

    def reset(self):
        """
        Pass reset function to universe environment.
        """
        return self.env.reset()

    def execute(self, action):
        """
        Pass action to universe environment, return reward, next step, terminal state and additional info.
        """
        pass_actions = []
        for action_name, value in action.items():
            if action_name == 'key':
                key_name = self._int_to_key(value)
                pass_actions.append(universe.spaces.KeyEvent.by_name(key_name, down=True))
            elif action_name == 'button':
                btn_name = self._int_to_btn(value)
                x, y = action.get('position', (0, 0))
                pass_actions.append(universe.spaces.PointerEvent(x, y, btn_name))

        state, reward, terminal, _ = self.env.step(pass_actions)
        return state, reward, terminal

    def _key_to_int(self, key_name):
        pass

    def _int_to_key(self, key_value):
        pass

    def _btn_to_int(self, key_name):
        pass

    def _int_to_btn(self, key_value):
        pass

    def configure(self, *args, **kwargs):
        self.env.configure(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    @property
    def states(self):
        print(self.env.observation_space)
        if isinstance(self.env.observation_space, VNCObservationSpace):
            reg = universe.runtime_spec('flashgames').server_registry
            return dict(
                vision=dict(type=float, shape=(reg[self.env_id]["width"], reg[self.env_id]["height"], 3))
                #text=dict(type=int, shape=(1,))
            )
        elif isinstance(self.env.observation_space, Discrete):
            return dict(shape=(), type='float')
        else:
            return dict(shape=tuple(self.env.observation_space.shape), type='float')

    @property
    def actions(self):
        if isinstance(self.env.action_space, VNCActionSpace):
            return dict(
                key=dict(continuous=False, num_actions=len(self.env.action_space.keys)),
                button=dict(continuous=False, num_actions=len(self.env.action_space.buttonmasks)),
                position=dict(continuous=False, num_actions=self.env.action_space.screen_shape[0] * self.env.action_space.screen_shape[1])
            )
        elif isinstance(self.env.action_space, Discrete):
            return dict(continuous=False, num_actions=self.env.action_space.n)
        elif len(self.env.action_space.shape) == 1:
            return {'action' + str(n): dict(continuous=True) for n in range(len(self.env.action_space.shape[0]))}
        else:
            raise TensorForceError()
