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

import unittest

from test.unittest_base import UnittestBase


class TestEnvironments(UnittestBase, unittest.TestCase):

    def test_ale(self):
        self.start_tests(name='ale')

    def test_maze_explorer(self):
        self.start_tests(name='maze-explorer')
        self.unittest(
            environment=dict(environment='mazeexp', level=0), num_episodes=2,
            max_episode_timesteps=100
        )

    def test_open_sim(self):
        self.start_tests(name='open-sim')
        self.unittest(environment=dict(environment='osim', level='Arm2D'), num_episodes=2)

    def test_openai_gym(self):
        self.start_tests(name='openai-gym')
        self.unittest(environment=dict(environment='gym', level='CartPole-v1'), num_episodes=2)

    def test_openai_retro(self):
        self.start_tests(name='openai-retro')
        self.unittest(
            environment=dict(environment='retro', level='Airstriker-Genesis'), num_episodes=2,
            max_episode_timesteps=100
        )

    def test_ple(self):
        self.start_tests(name='pygame-learning-environment')
        # self.unittest(
        #     environment=dict(environment='ple', level='Pong'), num_episodes=2,
        #     max_episode_timesteps=100
        # )

    def test_vizdoom(self):
        self.start_tests(name='vizdoom')
