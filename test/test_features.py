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

import os
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from tensorforce import Agent, Environment, Runner
from test.unittest_base import UnittestBase


class TestFeatures(UnittestBase, unittest.TestCase):

    # @pytest.mark.skip(reason='problems with processes/sockets in travis')
    def test_parallelization(self):
        self.start_tests(name='parallelization')

        agent = self.agent_spec(parallel_interactions=2)
        environment = self.environment_spec()

        runner = Runner(agent=agent, environment=environment, num_parallel=2)
        runner.run(num_episodes=5, use_tqdm=False)
        runner.close()
        self.finished_test()

        runner = Runner(
            agent=agent, environment=environment, num_parallel=2, remote='multiprocessing'
        )
        runner.run(num_episodes=5, use_tqdm=False)
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
        runner.run(num_episodes=5, use_tqdm=False)
        runner.close()
        server1.join()
        server2.join()
        self.finished_test()

    def test_act_experience_update(self):
        self.start_tests(name='act-experience-update')

        agent, environment = self.prepare(update=dict(unit='episodes', batch_size=1))

        for n in range(2):
            states = environment.reset()
            internals = agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = agent.act(states=states, internals=internals, independent=True)
                next_states, terminal, reward = environment.execute(actions=actions)
                agent.experience(
                    states=states, internals=internals, actions=actions, terminal=terminal,
                    reward=reward
                )
                states = next_states
            agent.update()

        self.finished_test()

    def test_pretrain(self):
        # FEATURES.MD
        self.start_tests(name='pretrain')

        with TemporaryDirectory() as directory:
            agent, environment = self.prepare(recorder=dict(directory=directory))

            for _ in range(3):
                states = environment.reset()
                terminal = False
                while not terminal:
                    actions = agent.act(states=states)
                    states, terminal, reward = environment.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)

            agent.close()

            # TODO: recorder currently does not include internal states
            agent = Agent.create(agent=self.agent_spec(
                policy=dict(network=dict(type='auto', size=8, depth=1, rnn=False))
            ), memory=10, environment=environment)

            agent.pretrain(directory=directory, num_iterations=2, num_traces=2, num_updates=3)

            agent.close()
            environment.close()

            files = os.listdir(path=directory)
            self.assertTrue(
                all(file.startswith('trace-') and file.endswith('.npz') for file in files)
            )

        self.finished_test()
