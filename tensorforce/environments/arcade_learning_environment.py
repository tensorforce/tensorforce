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

import numpy as np

from tensorforce.environments import Environment


class ArcadeLearningEnvironment(Environment):
    """
    [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
    adapter (specification key: `ale`, `arcade_learning_environment`).

    May require:
    ```bash
    sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake

    git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
    cd Arcade-Learning-Environment

    mkdir build && cd build
    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
    make -j 4
    cd ..

    pip3 install .
    ```

    Args:
        level (string): ALE rom file
            (<span style="color:#C00000"><b>required</b></span>).
        loss_of_life_termination: Signals a terminal state on loss of life
            (<span style="color:#00C000"><b>default</b></span>: false).
        loss_of_life_reward (float): Reward/Penalty on loss of life (negative values are a penalty)
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        repeat_action_probability (float): Repeats last action with given probability
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        visualize (bool): Whether to visualize interaction
            (<span style="color:#00C000"><b>default</b></span>: false).
        frame_skip (int > 0): Number of times to repeat an action without observing
            (<span style="color:#00C000"><b>default</b></span>: 1).
        seed (int): Random seed
            (<span style="color:#00C000"><b>default</b></span>: none).
    """

    def __init__(
        self, level, life_loss_terminal=False, life_loss_punishment=0.0,
        repeat_action_probability=0.0, visualize=False, frame_skip=1, seed=None
    ):
        super().__init__()

        from ale_python_interface import ALEInterface

        self.environment = ALEInterface()
        self.rom_file = level

        self.life_loss_terminal = life_loss_terminal
        self.life_loss_punishment = life_loss_punishment

        self.environment.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.environment.setBool(b'display_screen', visualize)
        self.environment.setInt(b'frame_skip', frame_skip)
        if seed is not None:
            self.environment.setInt(b'random_seed', seed)

        # All set commands must be done before loading the ROM.
        self.environment.loadROM(rom_file=self.rom_file.encode())
        self.available_actions = tuple(self.environment.getLegalActionSet())

        # Full list of actions:
        # No-Op, Fire, Up, Right, Left, Down, Up Right, Up Left, Down Right, Down Left, Up Fire,
        # Right Fire, Left Fire, Down Fire, Up Right Fire, Up Left Fire, Down Right Fire, Down Left
        # Fire

    def __str__(self):
        return super().__str__() + '({})'.format(self.rom_file)

    def states(self):
        width, height = self.environment.getScreenDims()
        return dict(type='float', shape=(height, width, 3))

    def actions(self):
        return dict(type='int', num_values=len(self.available_actions))

    def close(self):
        self.environment.__del__()
        self.environment = None

    def get_states(self):
        screen = np.copy(self.environment.getScreenRGB(screen_data=self.screen))
        screen = screen.astype(dtype=np.float32) / 255.0
        return screen

    def reset(self):
        self.environment.reset_game()
        width, height = self.environment.getScreenDims()
        self.screen = np.empty((height, width, 3), dtype=np.uint8)
        self.lives = self.environment.lives()
        return self.get_states()

    def execute(self, actions):
        reward = self.environment.act(action=self.available_actions[actions])
        terminal = self.environment.game_over()
        states = self.get_states()

        next_lives = self.environment.lives()
        if next_lives < self.lives:
            if self.life_loss_terminal:
                terminal = True
            elif self.life_loss_punishment > 0.0:
                reward -= self.life_loss_punishment
            self.lives = next_lives

        return states, terminal, reward
