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

import pytest
import unittest

from test.unittest_base import UnittestBase


class TestEnvironments(UnittestBase, unittest.TestCase):

    num_episodes = 2

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_ale(self):
        self.start_tests(name='ale')
        self.unittest(
            environment=dict(environment='ale', level='test/data/Breakout.bin'), num_episodes=2
        )

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_maze_explorer(self):
        self.start_tests(name='maze-explorer')
        self.unittest(environment=dict(environment='mazeexp', level=0), num_episodes=2)

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_open_sim(self):
        self.start_tests(name='open-sim')
        self.unittest(environment=dict(environment='osim', level='Arm2D'), num_episodes=2)

    def test_openai_gym(self):
        self.start_tests(name='openai-gym')
        self.unittest(environment=dict(environment='gym', level='CartPole-v0'), num_episodes=2)

        self.unittest(
            environment=dict(environment='gym', level='CartPole', max_episode_steps=False),
            num_episodes=2
        )

        from gym.envs.classic_control import CartPoleEnv

        self.unittest(environment=dict(environment='gym', level=CartPoleEnv()), num_episodes=2)

    def test_openai_retro(self):
        self.start_tests(name='openai-retro')
        self.unittest(
            environment=dict(environment='retro', level='Airstriker-Genesis'), num_episodes=2
        )

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_ple(self):
        self.start_tests(name='pygame-learning-environment')
        self.unittest(environment=dict(environment='ple', level='Pong'), num_episodes=2)

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_vizdoom(self):
        self.start_tests(name='vizdoom')
        self.unittest(
            environment=dict(environment='vizdoom', level='test/data/basic.cfg'), memory=1000
        )
