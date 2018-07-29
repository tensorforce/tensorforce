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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from ale_python_interface import ALEInterface

from tensorforce.environments import Environment


class ALE(Environment):
    """
    Arcade Learning Environment (ALE). https://github.com/mgbellemare/Arcade-Learning-Environment
    """

    def __init__(
        self,
        rom,
        frame_skip=1,
        repeat_action_probability=0.0,
        loss_of_life_termination=False,
        loss_of_life_reward=0,
        display_screen=False,
        seed=np.random.RandomState()
    ):
        """
        Initialize ALE.

        Args:
            rom: Rom filename and directory.
            frame_skip: Repeat action for n frames. Default 1.
            repeat_action_probability: Repeats last action with given probability. Default 0.
            loss_of_life_termination: Signals a terminal state on loss of life. Default False.
            loss_of_life_reward: Reward/Penalty on loss of life (negative values are a penalty). Default 0.
            display_screen: Displays the emulator screen. Default False.
            seed: Random seed
        """

        self.ale = ALEInterface()
        self.rom = rom

        self.ale.setBool(b'display_screen', display_screen)
        self.ale.setInt(b'random_seed', seed.randint(0, 9999))
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.setBool(b'color_averaging', False)
        self.ale.setInt(b'frame_skip', frame_skip)

        # All set commands must be done before loading the ROM.
        self.ale.loadROM(rom.encode())

        # Setup gamescreen object.
        width, height = self.ale.getScreenDims()
        self.gamescreen = np.empty((height, width, 3), dtype=np.uint8)

        self.frame_skip = frame_skip

        # Setup action converter.
        # ALE returns legal action indexes, convert these to just numbers.
        self.action_inds = self.ale.getMinimalActionSet()

        # Setup lives
        self.loss_of_life_reward = loss_of_life_reward
        self.cur_lives = self.ale.lives()
        self.loss_of_life_termination = loss_of_life_termination
        self.life_lost = False

    def __str__(self):
        return 'ALE({})'.format(self.rom)

    def close(self):
        self.ale = None

    def reset(self):
        self.ale.reset_game()
        self.cur_lives = self.ale.lives()
        self.life_lost = False
        # Clear gamescreen.
        self.gamescreen = np.empty(self.gamescreen.shape, dtype=np.uint8)
        return self.current_state

    def execute(self, action):
        # Convert action to ale action.
        ale_action = self.action_inds[action]

        # Get reward and process terminal & next state.
        rew = self.ale.act(ale_action)
        if self.loss_of_life_termination or self.loss_of_life_reward != 0:
            new_lives = self.ale.lives()
            if new_lives < self.cur_lives:
                self.cur_lives = new_lives
                self.life_lost = True
                rew += self.loss_of_life_reward

        terminal = self.is_terminal
        state_tp1 = self.current_state
        return state_tp1, terminal, rew

    @property
    def states(self):
        return dict(shape=self.gamescreen.shape, type=float)

    @property
    def actions(self):
        return dict(type='int', num_actions=len(self.action_inds))

    @property
    def current_state(self):
        self.gamescreen = self.ale.getScreenRGB(self.gamescreen)
        return np.copy(self.gamescreen)

    @property
    def is_terminal(self):
        if self.loss_of_life_termination and self.life_lost:
            return True
        else:
            return self.ale.game_over()

    @property
    def action_names(self):
        action_names = [
            'No-Op',
            'Fire',
            'Up',
            'Right',
            'Left',
            'Down',
            'Up Right',
            'Up Left',
            'Down Right',
            'Down Left',
            'Up Fire',
            'Right Fire',
            'Left Fire',
            'Down Fire',
            'Up Right Fire',
            'Up Left Fire',
            'Down Right Fire',
            'Down Left Fire'
        ]
        return np.asarray(action_names)[self.action_inds]
