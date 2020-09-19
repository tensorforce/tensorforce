# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

    agent = dict(agent='random')
    experience_update = False

    def test_ale(self):
        self.start_tests(name='ale')
        self.unittest(
            environment=dict(environment='ale', level='test/data/Breakout.bin'), num_episodes=2
        )

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_open_sim(self):
        self.start_tests(name='open-sim')
        self.unittest(environment=dict(environment='osim', level='Arm2D'), num_episodes=2)
        self.unittest(environment=dict(environment='osim', level='L2M2019'), num_episodes=2)
        self.unittest(environment=dict(environment='osim', level='LegacyArm'), num_episodes=2)
        self.unittest(environment=dict(environment='osim', level='LegacyRun'), num_episodes=2)

    def test_openai_gym(self):
        self.start_tests(name='openai-gym')

        # state: box, action: discrete
        self.unittest(environment=dict(environment='gym', level='CartPole-v0'), num_episodes=2)

        # state: discrete, action: box
        self.unittest(environment=dict(environment='gym', level='GuessingGame'), num_episodes=2)

        # state: discrete, action: tuple(discrete)
        from gym.envs.algorithmic import ReverseEnv
        self.unittest(environment=ReverseEnv, num_episodes=2)

        # state: tuple, action: discrete
        from gym.envs.toy_text import BlackjackEnv
        self.unittest(environment=BlackjackEnv(), num_episodes=2)

        # Classic control
        # above: self.unittest(environment='CartPole-v1', num_episodes=2)
        self.unittest(environment='MountainCar-v0', num_episodes=2)
        self.unittest(environment='MountainCarContinuous-v0', num_episodes=2)
        self.unittest(environment='Pendulum-v0', num_episodes=2)
        self.unittest(environment='Acrobot-v1', num_episodes=2)

        # Box2d
        self.unittest(environment='LunarLander-v2', num_episodes=2)
        self.unittest(environment='LunarLanderContinuous-v2', num_episodes=2)
        self.unittest(environment='BipedalWalker-v3', num_episodes=2)
        self.unittest(environment='BipedalWalkerHardcore-v3', num_episodes=2)
        # below: self.unittest(environment='CarRacing-v0', num_episodes=2)

        # Toy text
        # above: self.unittest(environment='Blackjack-v0', num_episodes=2)
        self.unittest(environment='KellyCoinflip-v0', num_episodes=2)
        # TODO: out-of-bounds problems!
        # self.unittest(environment=dict(
        #     environment='KellyCoinflipGeneralized-v0', clip_distributions=True
        # ), num_episodes=2)
        self.unittest(environment='FrozenLake-v0', num_episodes=2)
        self.unittest(environment='FrozenLake8x8-v0', num_episodes=2)
        self.unittest(environment='CliffWalking-v0', num_episodes=2)
        self.unittest(environment='NChain-v0', num_episodes=2)
        self.unittest(environment='Roulette-v0', num_episodes=2)
        self.unittest(environment='Taxi-v3', num_episodes=2)
        # above: self.unittest(environment='GuessingGame-v0', num_episodes=2)
        self.unittest(environment='HotterColder-v0', num_episodes=2)

        # Algorithmic
        self.unittest(environment='Copy-v0', num_episodes=2)
        self.unittest(environment='RepeatCopy-v0', num_episodes=2)
        self.unittest(environment='ReversedAddition-v0', num_episodes=2)
        self.unittest(environment='ReversedAddition3-v0', num_episodes=2)
        self.unittest(environment='DuplicatedInput-v0', num_episodes=2)
        # above: self.unittest(environment='Reverse-v0', num_episodes=2)

        # Unit test
        self.unittest(environment='CubeCrash-v0', num_episodes=2)
        self.unittest(environment='CubeCrashSparse-v0', num_episodes=2)
        self.unittest(environment='CubeCrashScreenBecomesBlack-v0', num_episodes=2)
        self.unittest(environment='MemorizeDigits-v0', num_episodes=2)

    @pytest.mark.skip(reason='requires virtual frame buffer xvfb-run')
    def test_openai_gym2(self):
        # state: box, action: box with non-uniform bounds
        # xvfb-run -s "-screen 0 1400x900x24" python -m unittest ...
        self.unittest(environment='CarRacing-v0', num_episodes=2)

    def test_openai_retro(self):
        self.start_tests(name='openai-retro')
        self.unittest(
            environment=dict(environment='retro', level='Airstriker-Genesis'), num_episodes=2
        )

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_ple(self):
        self.start_tests(name='pygame-learning-environment')
        self.unittest(environment=dict(environment='ple', level='Catcher'), num_episodes=2)
        # Assets missing:
        # self.unittest(environment=dict(environment='ple', level='FlappyBird'), num_episodes=2)
        # self.unittest(environment=dict(environment='ple', level='MonsterKong'), num_episodes=2)
        self.unittest(environment=dict(environment='ple', level='Pixelcopter'), num_episodes=2)
        self.unittest(environment=dict(environment='ple', level='Pong'), num_episodes=2)
        self.unittest(environment=dict(environment='ple', level='PuckWorld'), num_episodes=2)
        self.unittest(environment=dict(environment='ple', level='RaycastMaze'), num_episodes=2)
        self.unittest(environment=dict(environment='ple', level='Snake'), num_episodes=2)
        self.unittest(environment=dict(environment='ple', level='WaterWorld'), num_episodes=2)

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_vizdoom(self):
        self.start_tests(name='vizdoom')
        self.unittest(
            environment=dict(environment='vizdoom', level='test/data/basic.cfg'), num_episodes=2
        )
