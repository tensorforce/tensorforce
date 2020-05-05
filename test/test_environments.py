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
from threading import Thread
import unittest

from tensorforce import Environment, Runner

from test.unittest_base import UnittestBase


class TestEnvironments(UnittestBase, unittest.TestCase):

    num_episodes = 2

    @pytest.mark.skip(reason='problems with processes/sockets in travis')
    def test_remote_environments(self):
        self.start_tests(name='remote-environments')

        agent = self.agent_spec(
            require_observe=True, update=dict(unit='episodes', batch_size=1),
            parallel_interactions=2
        )
        environment = self.environment_spec()

        runner = Runner(
            agent=agent, environment=environment, num_parallel=2, remote='multiprocessing'
        )
        runner.run(num_episodes=self.__class__.num_episodes, use_tqdm=False)
        runner.close()
        self.finished_test()

        def server(port):
            Environment.create(environment=environment, remote='socket-server', port=port)

        server1 = Thread(target=server, kwargs=dict(port=65432))
        server2 = Thread(target=server, kwargs=dict(port=65433))
        server1.start()
        server2.start()
        runner = Runner(
            agent=agent, num_parallel=2, remote='socket-client', host='127.0.0.1', port=65432
        )
        runner.run(num_episodes=self.__class__.num_episodes, use_tqdm=False)
        runner.close()
        server1.join()
        server2.join()

        self.finished_test()

    # @pytest.mark.skip(reason='not installed as part of travis')
    # def test_ale(self):
    #     self.start_tests(name='ale')
    #     self.unittest(
    #         environment=dict(environment='ale', level='test/data/Breakout.bin'), num_episodes=2
    #     )

    # @pytest.mark.skip(reason='not installed as part of travis')
    # def test_maze_explorer(self):
    #     self.start_tests(name='maze-explorer')
    #     self.unittest(environment=dict(environment='mazeexp', level=0))

    # @pytest.mark.skip(reason='not installed as part of travis')
    # def test_open_sim(self):
    #     self.start_tests(name='open-sim')
    #     self.unittest(environment=dict(environment='osim', level='Arm2D'))

    def test_openai_gym(self):
        self.start_tests(name='openai-gym')

        # state: box, action: discrete
        self.unittest(environment=dict(environment='gym', level='CartPole-v0'))

        # state: discrete, action: box
        self.unittest(
            environment=dict(environment='gym', level='GuessingGame', max_episode_steps=False)
        )

        # state: discrete, action: tuple(discrete)
        from gym.envs.algorithmic import ReverseEnv
        self.unittest(environment=ReverseEnv)

        # state: tuple, action: discrete
        from gym.envs.toy_text import BlackjackEnv
        self.unittest(environment=BlackjackEnv())

    @pytest.mark.skip(reason='breaks / takes too long')
    def test_openai_gym2(self):
        # state: box, action: box with non-uniform bounds
        # xvfb-run -s "-screen 0 1400x900x24" python -m unittest ...
        self.unittest(environment='CarRacing-v0')

        # Classic control
        self.unittest(environment='CartPole-v1')
        self.unittest(environment='MountainCar-v0')
        self.unittest(environment='MountainCarContinuous-v0')
        self.unittest(environment='Pendulum-v0')
        self.unittest(environment='Acrobot-v1')

        # Box2d
        self.unittest(environment='LunarLander-v2')
        self.unittest(environment='LunarLanderContinuous-v2')
        self.unittest(environment='BipedalWalker-v3')
        self.unittest(environment='BipedalWalkerHardcore-v3')
        # above: self.unittest(environment='CarRacing-v0')

        # Toy text
        # above: self.unittest(environment='Blackjack-v0')
        self.unittest(environment='KellyCoinflip-v0')
        self.unittest(environment='KellyCoinflipGeneralized-v0')
        self.unittest(environment='FrozenLake-v0')
        self.unittest(environment='FrozenLake8x8-v0')
        self.unittest(environment='CliffWalking-v0')
        self.unittest(environment='NChain-v0')
        self.unittest(environment='Roulette-v0')
        self.unittest(environment='Taxi-v3')
        self.unittest(environment='GuessingGame-v0')
        self.unittest(environment='HotterColder-v0')

        # Algorithmic
        self.unittest(environment='Copy-v0')
        self.unittest(environment='RepeatCopy-v0')
        self.unittest(environment='ReversedAddition-v0')
        self.unittest(environment='ReversedAddition3-v0')
        self.unittest(environment='DuplicatedInput-v0')
        # above: self.unittest(environment='Reverse-v0')

        # Unit test
        self.unittest(environment='CubeCrash-v0')
        self.unittest(environment='CubeCrashSparse-v0')
        self.unittest(environment='CubeCrashScreenBecomesBlack-v0')
        self.unittest(environment='MemorizeDigits-v0')

    def test_openai_retro(self):
        self.start_tests(name='openai-retro')
        self.unittest(environment=dict(environment='retro', level='Airstriker-Genesis'))

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_ple(self):
        self.start_tests(name='pygame-learning-environment')
        self.unittest(environment=dict(environment='ple', level='Pong'))

    @pytest.mark.skip(reason='not installed as part of travis')
    def test_vizdoom(self):
        self.start_tests(name='vizdoom')
        self.unittest(
            environment=dict(environment='vizdoom', level='test/data/basic.cfg'), memory=1000
        )
