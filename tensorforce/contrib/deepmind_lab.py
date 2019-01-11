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
import deepmind_lab
from tensorforce.environments import Environment


# TODO this has not been tested since 0.3 - potentially deprecated API
class DeepMindLab(Environment):
    """
    DeepMind Lab Integration:
    https://arxiv.org/abs/1612.03801
    https://github.com/deepmind/lab

    Since DeepMind lab is only available as source code, a manual install
    via bazel is required. Further, due to the way bazel handles external
    dependencies, cloning TensorForce into lab is the most convenient way to
    run it using the bazel BUILD file we provide. To use lab, first download
    and install it according to instructions
    <https://github.com/deepmind/lab/blob/master/docs/build.md>:

    ```bash
    git clone https://github.com/deepmind/lab.git
    ```

    Add to the lab main BUILD file:

    ```
    package(default_visibility = ["//visibility:public"])
    ```

    Clone TensorForce into the lab directory, then run the TensorForce bazel runner.

    Note that using any specific configuration file currently requires changing the Tensorforce
    BUILD file to adjust environment parameters.

    ```bash
    bazel run //tensorforce:lab_runner
    ```

    Please note that we have not tried to reproduce any lab results yet, and
    these instructions just explain connectivity in case someone wants to
    get started there.


    """

    def __init__(
        self,
        level_id,
        repeat_action=1,
        state_attribute='RGB_INTERLACED',
        settings={'width': '320', 'height': '240', 'fps': '60', 'appendCommand': ''}
    ):
        """
        Initialize DeepMind Lab environment.

        Args:
            level_id: string with id/descriptor of the level, e.g. 'seekavoid_arena_01'.
            repeat_action: number of frames the environment is advanced, executing the given action during every frame.
            state_attribute: Attributes which represents the state for this environment, should adhere to the
                specification given in DeepMindLabEnvironment.state_spec(level_id).
            settings: dict specifying additional settings as key-value string pairs. The following options
                are recognized: 'width' (horizontal resolution of the observation frames), 'height'
                (vertical resolution of the observation frames), 'fps' (frames per second) and 'appendCommand'
                (commands for the internal Quake console).

        """
        self.level_id = level_id
        self.level = deepmind_lab.Lab(level=level_id, observations=[state_attribute], config=settings)
        self.repeat_action = repeat_action
        self.state_attribute = state_attribute

    def __str__(self):
        return 'DeepMindLab({})'.format(self.level_id)

    def close(self):
        """
        Closes the environment and releases the underlying Quake III Arena instance.
        No other method calls possible afterwards.
        """
        self.level.close()
        self.level = None

    def reset(self):
        """
        Resets the environment to its initialization state. This method needs to be called to start a
        new episode after the last episode ended.

        :return: initial state
        """
        self.level.reset()  # optional: episode=-1, seed=None
        return self.level.observations()[self.state_attribute]

    def execute(self, action):
        """
        Pass action to universe environment, return reward, next step, terminal state and
        additional info.

        :param action: action to execute as numpy array, should have dtype np.intc and should adhere to
            the specification given in DeepMindLabEnvironment.action_spec(level_id)
        :return: dict containing the next state, the reward, and a boolean indicating if the
            next state is a terminal state
        """
        adjusted_action = list()
        for action_spec in self.level.action_spec():
            if action_spec['min'] == -1 and action_spec['max'] == 1:
                adjusted_action.append(action[action_spec['name']] - 1)
            else:
                adjusted_action.append(action[action_spec['name']])  # clip?
        action = np.array(adjusted_action, dtype=np.intc)

        reward = self.level.step(action=action, num_steps=self.repeat_action)
        state = self.level.observations()['RGB_INTERLACED']
        terminal = not self.level.is_running()
        return state, terminal, reward

    def states(self):
        states = dict()

        for state in self.level.observation_spec():
            state_type = state['dtype']

            if state_type == np.uint8:
                state_type = np.float32

            if state['name'] == self.state_attribute:
                return dict(shape=state['shape'], type=state_type)

        return states

    def actions(self):
        actions = dict()
        for action in self.level.action_spec():
            if action['min'] == -1 and action['max'] == 1:
                actions[action['name']] = dict(type='int', num_actions=3)
            else:
                actions[action['name']] = dict(type='float', min_value=action['min'], max_value=action['max'])
        return actions

    @property
    def num_steps(self):
        """
        Number of frames since the last reset() call.
        """
        return self.level.num_steps()

    @property
    def fps(self):
        """
        An advisory metric that correlates discrete environment steps ("frames") with real
        (wallclock) time: the number of frames per (real) second.
        """
        return self.level.fps()
