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
DeepMind Lab Integration:
https://arxiv.org/abs/1612.03801
https://github.com/deepmind/lab
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import deepmind_lab
from tensorforce.environments.environment import Environment


class DeepMindLabEnvironment(Environment):
    @staticmethod
    def state_spec(level_id):
        """
        Returns a list of dicts with keys 'dtype', 'shape' and 'name', specifying the available observations this DeepMind Lab environment supports.

        :param level_id: string with id/descriptor of the level
        """
        level = deepmind_lab.Lab(level_id, ())
        return level.observation_spec()

    @staticmethod
    def action_spec(level_id):
        """
        Returns a list of dicts with keys 'min', 'max' and 'name', specifying the shape of the actions expected by this DeepMind Lab environment.

        :param level_id: string with id/descriptor of the level
        """
        level = deepmind_lab.Lab(level_id, ())
        return level.action_spec()

    def __init__(self, level_id, num_steps=1, state_attributes=['RGB_INTERLACED'],
                 settings={'width': '320', 'height': '240', 'fps': '60', 'appendCommand': ''}):
        """
        Initialize DeepMind Lab environment.

        :param level_id: string with id/descriptor of the level, e.g. 'seekavoid_arena_01'
        :param num_steps: number of frames the environment is advanced, executing the given action during every frame
        :param state_attributes: list of attributes which represent the state for this environment, should adhere to the specification given in DeepMindLabEnvironment.state_spec(level_id)
        :param settings: dict specifying additional settings as key-value string pairs. The following options are recognized: 'width' (horizontal resolution of the observation frames), 'height' (vertical resolution of the observation frames), 'fps' (frames per second) and 'appendCommand' (commands for the internal Quake console).
        """

        #TODO use state attributes as a key to fetch the right state from the state list
        # whenever state is accessed
        self.level_id = level_id
        self.level = deepmind_lab.Lab(level=level_id, observations=state_attributes, config=settings)
        self._num_steps = num_steps

    def reset(self):
        """
        Resets the environment to its initialization state. This method needs to be called to start a new episode after the last episode ended.

        :return: initial state
        """
        self.level.reset()  # optional: episode=-1, seed=None
        return self.level.observations()['RGB_INTERLACED']

    def close(self):
        """
        Closes the environment and releases the underlying Quake III Arena instance. No other method calls possible afterwards.
        """
        self.level.close()
        self.level = None

    def execute_action(self, action):
        """
        Pass action to universe environment, return reward, next step, terminal state and additional info.

        :param action: action to execute as numpy array, should have dtype np.intc and should adhere to the specification given in DeepMindLabEnvironment.action_spec(level_id)
        :return: dict containing the next state, the reward, and a boolean indicating if the next state is a terminal state
        """

        #TODO this is not sensible at all for learning and just here to test lab connectivity
        #TODO this will be replaced by a more general action representation
        # Lab uses named action dicts while gym/universe use simple ints/arrays, so we
        # will add wrapper
        action_mask = np.zeros(len(self.level.action_spec()), dtype=np.intc)
        action_mask[action] = 1
        reward = self.level.step(action=action_mask, num_steps=self._num_steps)

        state = self.level.observations()['RGB_INTERLACED']
        terminal_state = self.level.is_running()

        return dict(state=state, reward=reward, terminal_state=terminal_state)

    @property
    def actions(self):
        return len(self.level.action_spec())

    @property
    def action_shape(self):
        #TODO also just for demonstrating basic lab connectivity
        return []
#       return [len(self.level.action_spec())]

    @property
    def state_shape(self):
        # TODO this is just for demonstration, we need a more general way of
        # modelling states
        return self.level.observation_spec()[0]['shape']

    @property
    def num_steps(self):
        """
        Number of frames since the last reset() call.
        """
        return self.level.num_steps()

    @property
    def fps(self):
        """
        An advisory metric that correlates discrete environment steps ("frames") with real (wallclock) time: the number of frames per (real) second.
        """
        return self.level.fps()
