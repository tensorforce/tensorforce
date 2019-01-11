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

"""
Deepmind Pycolab Integration: https://github.com/deepmind/pycolab.
"""

import numpy as np
import copy
from tensorforce import TensorForceError
from tensorforce.environments import Environment


class DMPycolab(Environment):
    """
    Bindings for Deepmind Pycolab environment https://github.com/deepmind/pycolab
    """

    def __init__(self, game, ui, visualize=False):
        """
        Initialize Pycolab environment.

        Args:
            game: Pycolab Game Engine object. See https://github.com/deepmind/pycolab/tree/master/pycolab/examples
            ui: Pycolab CursesUI object. See https://github.com/deepmind/pycolab/tree/master/pycolab/examples
            visualize: If set True, the program will visualize the trainings of Pycolab game # TODO
        """
        self.game = game
        self.init_game = copy.deepcopy(self.game)
        self.ui = ui
        self.visualize = visualize

        first_obs, first_reward, _ = self.game.its_showtime()
        self._actions = DMPycolab.get_action_space(self.ui)
        self._states = DMPycolab.get_state_space(first_obs, self.ui._croppers)


    def __str__(self):
        return 'DeepMind Pycolab({})'.format(self.game)

    def states(self):
        return self._states

    def actions(self):
        return self._actions

    def close(self):
        self.game._the_plot._clear_engine_directives()
        self.game = None

    def reset(self):
        self.game = copy.deepcopy(self.init_game)
        first_obs, first_reward, _ = self.game.its_showtime()
        return DMPycolab.crop_and_flatten(first_obs, self.ui._croppers)

    def execute(self, action):
        observation, reward, _ = self.game.play(action)
        observations = DMPycolab.crop_and_flatten(observation, self.ui._croppers)

        terminal = self.game.game_over
        if reward is None:
            reward = 0
        return observations, terminal, reward

    @staticmethod
    def get_action_space(ui):
        return dict(type='int', num_actions=len(ui._keycodes_to_actions.values()))

    @staticmethod
    def get_state_space(first_obs, croppers):
        obs = DMPycolab.crop_and_flatten(first_obs, croppers)
        return dict(shape=obs.shape, type='int')

    @staticmethod
    def crop_and_flatten(observation, croppers):
        observations = [cropper.crop(observation) for cropper in croppers]
        flat_obs = np.array([obs.board for obs in observations]).flatten()
        return flat_obs
