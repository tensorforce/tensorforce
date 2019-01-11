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

import time

from tensorforce import TensorForceError
from tensorforce.environments import Environment

import ple
from ple.games import *

class PLE(Environment):
    """
    PyGame-Learning-Environment integration -
    https://github.com/ntasfi/PyGame-Learning-Environment/
    """

    def __init__(self, env_id, visualize=False):
        self.env_id = env_id
        self.visualize = visualize
        env_dict = {
            "doom" : Doom,
            "flappybird" : FlappyBird,
            "monsterkong" : MonsterKong,
            "catcher" : Catcher,
            "pixelcopter" : Pixelcopter,
            "pong" : Pong,
            "puckworld" : PuckWorld,
            "raycastmaze" : RaycastMaze,
            "snake" : Snake,
            "waterworld" : WaterWorld
        }
        try:
            # Maybe try to implement for python 2.7? Definitely deprecated for 3.6
            if self.env_id == "doom":
                raise TensorForceError("Doom-Py Deprecated")
            else:
                self.game = env_dict[env_id]()
                self.env = ple.PLE(self.game, display_screen=visualize)
        except KeyError:
            print('Game not implemented in PyGame-Learning-Environemnt or these bindings')
            print('Implemented environments include:')
            print('"flappybird", "monsterkong", "catcher"')
            print('"pixelcopter", "pong", "puckworld", "waterworld"')

    def __str__(self):
        return 'pygame_learning_environment({})'.format(self.env_id)

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        self.env = None

    def seed(self, seed): # pylint: disable=E0202
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        if seed is None:
            self.env.seed = round(time.time())
        else:
            self.env.seed = seed
        return self.env.seed

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        self.env.reset_game()
        return self.env.getScreenRGB()

    def execute(self, action):
        """
        Executes action, observes next state and reward.

        Args:
            actions: Action to execute.

        Returns:
            (Dict of) next state(s), boolean indicating terminal, and reward signal.
        """
        if self.env.game_over():
            return self.env.getScreenRGB(), True, 0

        action_space = self.env.getActionSet()
        reward = self.env.act(action_space[action])
        new_state = self.env.getScreenRGB()
        done = self.env.game_over()
        return new_state, done, reward

    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are
        available simultaneously.

        Returns: dict of state properties (shape and type).

        """
        screen = self.env.getScreenRGB()
        return dict(shape=screen.shape, type='int')

    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are
        available simultaneously.

        Returns: dict of action properties (continuous, number of actions)

        """
        return dict(num_actions=len(self.env.getActionSet()), type='int')
