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

import unittest
import numpy as np
from tensorforce import Configuration
from tensorforce.core.memories.replay import Replay


class TestReplayMemory(unittest.TestCase):
    @staticmethod
    def init_memory(random_sampling):
        states_dict = {
            "state1": {
                "shape": (2, 2),
                "type": 'float'
            },
            "state2": {
                "shape": [10],
                "type": 'float'
            }
        }
        states_config = Configuration(**states_dict)
        actions_dict = {
            "action1": {
                "shape": [1],
                "continuous": False,
            },
            "action2": {
                "shape": [4],
                "continuous": True,
            }

        }
        actions_config = Configuration(**actions_dict)
        replay = Replay(10, states_config, actions_config, random_sampling=random_sampling)
        return replay

    @staticmethod
    def add_sample(replay, sample_ind):
        state1 = np.ones((2, 2)) * sample_ind
        state2 = np.ones((10,)) * sample_ind
        action1 = np.ones((1,)) * sample_ind
        action2 = np.ones((4,)) * sample_ind
        states = {"state1": state1, "state2": state2}
        actions = {"action1": action1, "action2": action2}
        reward = sample_ind
        terminal = sample_ind % 2 == 0 and sample_ind > 0
        replay.add_observation(states, actions, reward, terminal, [])

    def test_setup(self):
        replay = self.init_memory(False)
        replay = self.init_memory(True)

    def test_add_observation(self):
        replay = self.init_memory(False)
        self.add_sample(replay, 0)
        self.add_sample(replay, 1)

    def test_get_sequential_batch(self):
        replay = self.init_memory(False)
        for i in range(15):
            self.add_sample(replay, i)

        # make sure terminals are only in the last slot
        # otherwise the agent is getting a state from the last episode
        for i in range(50):
            batch = replay.get_batch(2)
            assert np.all(batch['terminals'][:-1] == False), "sequential get_batch returned terminal in a position other than the last"
