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
import itertools

import numpy as np

from tensorforce.environments import Environment


class ViZDoom(Environment):
    """
    [ViZDoom](https://github.com/mwydmuch/ViZDoom) environment adapter (specification key:
    `vizdoom`).

    May require:
    ```bash
    sudo apt-get install g++ build-essential libsdl2-dev zlib1g-dev libmpg123-dev libjpeg-dev \
    libsndfile1-dev nasm tar libbz2-dev libgtk2.0-dev make cmake git chrpath timidity \
    libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev \
    liblua5.1-dev

    pip3 install vizdoom
    ```

    Args:
        level (string): ViZDoom configuration file
            (<span style="color:#C00000"><b>required</b></span>).
        include_variables (bool): Whether to include game variables to state
            (<span style="color:#00C000"><b>default</b></span>: false).
        factored_action (bool): Whether to use factored action representation
            (<span style="color:#00C000"><b>default</b></span>: false).
        visualize (bool): Whether to visualize interaction
            (<span style="color:#00C000"><b>default</b></span>: false).
        frame_skip (int > 0): Number of times to repeat an action without observing
            (<span style="color:#00C000"><b>default</b></span>: 12).
        seed (int): Random seed
            (<span style="color:#00C000"><b>default</b></span>: none).
    """

    def __init__(
        self, level, visualize=False, include_variables=False, factored_action=False,
        frame_skip=12, seed=None
    ):
        super().__init__()

        from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

        self.config_file = level
        self.include_variables = include_variables
        self.factored_action = factored_action
        self.visualize = visualize
        self.frame_skip = frame_skip

        self.environment = DoomGame()
        self.environment.load_config(self.config_file)
        if self.visualize:
            self.environment.set_window_visible(True)
            self.environment.set_mode(Mode.ASYNC_PLAYER)
        else:
            self.environment.set_window_visible(False)
            self.environment.set_mode(Mode.PLAYER)
        # e.g. CRCGCB, RGB24, GRAY8
        self.environment.set_screen_format(ScreenFormat.RGB24)
        # e.g. RES_320X240, RES_640X480, RES_1920X1080
        self.environment.set_screen_resolution(ScreenResolution.RES_640X480)
        self.environment.set_depth_buffer_enabled(False)
        self.environment.set_labels_buffer_enabled(False)
        self.environment.set_automap_buffer_enabled(False)
        if seed is not None:
            self.environment.setSeed(seed)
        self.environment.init()

        self.state_shape = (480, 640, 3)
        self.num_variables = self.environment.get_available_game_variables_size()
        self.num_buttons = self.environment.get_available_buttons_size()
        self.available_actions = [
            tuple(a) for a in itertools.product([0, 1], repeat=self.num_buttons)
        ]

    def __str__(self):
        return super().__str__() + '({})'.format(self.config_file)

    def states(self):
        if self.include_variables:
            return OrderedDict(
                screen=dict(type='float', shape=self.state_shape),
                variables=dict(type='float', shape=self.num_variables)
            )
        else:
            return dict(type='float', shape=self.state_shape)

    def actions(self):
        if self.factored_action:
            return dict(type='bool', shape=self.num_buttons)
        else:
            return dict(type='int', shape=(), num_values=len(self.available_actions))

    def close(self):
        self.environment.close()
        self.environment = None

    def get_states(self):
        state = self.environment.get_state()
        screen = state.screen_buffer.astype(dtype=np.float32) / 255.0
        if self.include_variables:
            return OrderedDict(screen=screen, variables=state.game_variables)
        else:
            return screen

    def reset(self):
        self.environment.new_episode()
        self.current_states = self.get_states()
        return self.current_states

    def execute(self, actions):
        if self.factored_action:
            action = np.where(actions, 1.0, 0.0)
        else:
            action = self.available_actions[actions]
        if self.visualize:
            self.environment.set_action(action)
            reward = 0.0
            for _ in range(self.frame_skip):
                self.environment.advance_action()
                reward += self.environment.get_last_reward()
        else:
            reward = self.environment.make_action(list(action), self.frame_skip)
        terminal = self.environment.is_episode_finished()
        if not terminal:
            self.current_states = self.get_states()
        return self.current_states, terminal, reward
