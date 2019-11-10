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

from collections import OrderedDict
import os

import numpy as np

from tensorforce import TensorforceError
from tensorforce.environments import Environment


class PyGameLearningEnvironment(Environment):
    """
    [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment/)
    environment adapter (specification key: `ple`, `pygame_learning_environment`).

    May require:
    ```bash
    sudo apt-get install git python3-dev python3-setuptools python3-numpy python3-opengl \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev \
    libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libtiff5-dev libx11-6 \
    libx11-dev fluid-soundfont-gm timgm6mb-soundfont xfonts-base xfonts-100dpi xfonts-75dpi \
    xfonts-cyrillic fontconfig fonts-freefont-ttf libfreetype6-dev

    pip3 install git+https://github.com/pygame/pygame.git

    pip3 install git+https://github.com/ntasfi/PyGame-Learning-Environment.git
    ```

    Args:
        level (string | subclass of `ple.games.base`): Game instance or name of class in
            `ple.games`, like "Catcher", "Doom", "FlappyBird", "MonsterKong", "Pixelcopter", 
            "Pong", "PuckWorld", "RaycastMaze", "Snake", "WaterWorld"
            (<span style="color:#C00000"><b>required</b></span>).
        visualize (bool): Whether to visualize interaction
            (<span style="color:#00C000"><b>default</b></span>: false).
        frame_skip (int > 0): Number of times to repeat an action without observing
            (<span style="color:#00C000"><b>default</b></span>: 1).
        fps (int > 0): The desired frames per second we want to run our game at
            (<span style="color:#00C000"><b>default</b></span>: 30).
    """

    @classmethod
    def levels(cls):
        import ple

        levels = list()
        for level in dir(ple.games):
            level_cls = getattr(ple.games, level)
            if isinstance(level_cls, type) and issubclass(level_cls, ple.games.base.PyGameWrapper):
                levels.append(level)
        return levels

    def __init__(self, level, visualize=False, frame_skip=1, fps=30):
        super().__init__()

        import ple

        if isinstance(level, str):
            assert level in PyGameLearningEnvironment.levels()
            level = getattr(ple.games, level)()

        if not visualize:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        self.environment = ple.PLE(
            game=level, fps=fps, frame_skip=frame_skip, display_screen=visualize
            # num_steps=1, reward_values={}, force_fps=True, add_noop_action=True, NOOP=K_F15,
            # state_preprocessor=None, rng=24
        )
        self.environment.init()

        self.has_game_state = self.environment.getGameStateDims() is not None
        self.available_actions = tuple(self.environment.getActionSet())

    def __str__(self):
        return super().__str__() + '({})'.format(self.environment.__class__.__name__)

    def states(self):
        if self.has_game_state:
            return OrderedDict(
                screen=dict(type='float', shape=(tuple(self.environment.getScreenDims()) + (3,))),
                state=dict(type='float', shape=(tuple(self.environment.getGameStateDims) + (3,)))
            )
        else:
            return dict(type='float', shape=(tuple(self.environment.getScreenDims()) + (3,)))

    def actions(self):
        return dict(type='int', shape=(), num_values=len(self.available_actions))

    def close(self):
        self.environment = None

    def get_states(self):
        screen = self.environment.getScreenRGB().astype(dtype=np.float32) / 255.0
        if self.has_game_state:
            return OrderedDict(screen=screen, state=self.environment.getGameState())
        else:
            return screen

    def reset(self):
        self.environment.reset_game()
        return self.get_states()

    def execute(self, actions):
        if self.environment.game_over():
            raise TensorforceError.unexpected()
        reward = self.environment.act(action=self.available_actions[actions])
        terminal = self.environment.game_over()
        states = self.get_states()
        return states, terminal, reward
