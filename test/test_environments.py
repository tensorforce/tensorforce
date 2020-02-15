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
from test.unittest_environment import UnittestEnvironment


class TestEnvironments(UnittestBase, unittest.TestCase):

    num_episodes = 2

    @pytest.mark.skip(reason='problems with processes/sockets in travis')
    def test_remote_environments(self):
        self.start_tests(name='remote-environments')

        agent, _ = self.prepare(
            require_observe=True, update=dict(unit='episodes', batch_size=1),
            parallel_interactions=2
        )
        environment = dict(
            environment=UnittestEnvironment, states=self.__class__.states,
            actions=self.__class__.actions, min_timesteps=self.__class__.min_timesteps
        )

        runner = Runner(
            agent=agent, environment=environment, num_parallel=2, max_episode_timesteps=5,
            remote='multiprocessing'
        )
        runner.run(num_episodes=self.__class__.num_episodes, use_tqdm=False)
        runner.close()
        self.finished_test()

        def server(port):
            Environment.create(
                environment=environment, max_episode_timesteps=5, remote='socket-server',
                port=port, states=self.__class__.states, actions=self.__class__.actions,
                min_timesteps=self.__class__.min_timesteps
            )

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

        agent.close()
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
