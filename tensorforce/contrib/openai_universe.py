# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

import gym
import universe
from gym.spaces.discrete import Discrete
from universe.spaces import VNCActionSpace, VNCObservationSpace

from tensorforce import TensorForceError
from tensorforce.environments import Environment


class OpenAIUniverse(Environment):
    """
    OpenAI Universe Integration: https://universe.openai.com/.
    Contains OpenAI Gym: https://gym.openai.com/.
    """

    def __init__(self, env_id):
        """
        Initialize OpenAI universe environment.

        Args:
            env_id: string with id/descriptor of the universe environment, e.g. 'HarvestDay-v0'.
        """
        self.env_id = env_id
        self.env = gym.make(env_id)

    def __str__(self):
        return 'OpenAI-Universe({})'.format(self.env_id)

    def close(self):
        self.env = None

    def reset(self):
        state = self.env.reset()
        if state == [None]:
            state, r, t = self._wait_state(state, None, None)

        if isinstance(state[0], dict):
            # We can't handle string states right now, so omit the text state for now
            state[0].pop('text', None)

        return state[0]

    def execute(self, action):
        state, terminal, reward = self._execute(action)
        return self._wait_state(state, terminal, reward)

    #TODO fix this for single actions (np array). 
    def _execute(self, actions):
        pass_actions = []
        for action_name, value in actions.items():
            if action_name == 'key':
                key_event = self._int_to_key(value)
                pass_actions.append(key_event)
            elif action_name == 'button':
                btn_event = self._int_to_btn(value)
                x, y = self._int_to_pos(actions.get('position', 0))
                pass_actions.append(universe.spaces.PointerEvent(x, y, btn_event))

        state, reward, terminal, _ = self.env.step([pass_actions])

        if isinstance(state[0], dict):
            # We can't handle string states right now, so omit the text state for now
            state[0].pop('text', None)

        return state[0], terminal[0], reward[0]

    def _int_to_pos(self, flat_position):
        """Returns x, y from flat_position integer.

        Args:
            flat_position: flattened position integer

        Returns: x, y

        """
        return flat_position % self.env.action_space.screen_shape[0],\
            flat_position % self.env.action_space.screen_shape[1]

    def _key_to_int(self, key_event):
        return self.env.action_space.keys.index(key_event)

    def _int_to_key(self, key_value):
        return self.env.action_space.keys[key_value]

    def _btn_to_int(self, btn_event):
        return self.env.action_space.buttonmasks.index(btn_event)

    def _int_to_btn(self, btn_value):
        return self.env.action_space.buttonmasks[btn_value]

    def _wait_state(self, state, reward, terminal):
        """
        Wait until there is a state.
        """
        while state == [None] or not state:
             state, terminal, reward = self._execute(dict(key=0))

        return state, terminal, reward

    def configure(self, *args, **kwargs):
        self.env.configure(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def states(self):
        print(self.env.observation_space)
        if isinstance(self.env.observation_space, VNCObservationSpace):
            return dict(
                # VNCObeservationSpace seems to be hardcoded to 1024x768
                vision=dict(type='float', shape=(768, 1024, 3))
                # vision = dict(type=float, shape=(self.env.action_space.screen_shape[1],
                #  self.env.action_space.screen_shape[0], 3))
                # text=dict(type=str, shape=(1,)) # TODO: implement string states
            )
        elif isinstance(self.env.observation_space, Discrete):
            return dict(shape=(), type='float')
        else:
            return dict(shape=tuple(self.env.observation_space.shape), type='float')

    def actions(self):
        if isinstance(self.env.action_space, VNCActionSpace):
            return dict(
                key=dict(type='int', num_actions=len(self.env.action_space.keys)),
                button=dict(type='int', num_actions=len(self.env.action_space.buttonmasks)),
                position=dict(
                    type='int',
                    num_actions=self.env.action_space.screen_shape[0] * self.env.action_space.screen_shape[1]
                )
            )
        elif isinstance(self.env.action_space, Discrete):
            return dict(type='int', num_actions=self.env.action_space.n)
        elif len(self.env.action_space.shape) == 1:
            return {'action' + str(n): dict(type='float') for n in range(len(self.env.action_space.shape[0]))}
        else:
            raise TensorForceError()
