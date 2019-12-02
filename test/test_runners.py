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

import copy
import time
import unittest

from tensorforce.execution import ParallelRunner, Runner
from test.unittest_base import UnittestBase


class TestRunners(UnittestBase, unittest.TestCase):

    require_observe = True

    def test_runner(self):
        self.start_tests(name='runner')

        agent, environment = self.prepare()

        runner = Runner(agent=agent, environment=environment)
        runner.run(num_episodes=10, use_tqdm=False)
        runner.close()

        self.finished_test()

        # callback
        agent, environment = self.prepare()

        runner = Runner(agent=agent, environment=environment)

        callback_episode_frequency = 2
        self.num_callbacks = 0

        def callback(r):
            self.num_callbacks += 1
            self.assertEqual(r.episodes, self.num_callbacks * callback_episode_frequency)

        runner.run(
            num_episodes=5, callback=callback,
            callback_episode_frequency=callback_episode_frequency, use_tqdm=False
        )

        callback_timestep_frequency = 3
        self.num_callbacks = 0

        def callback(r):
            self.num_callbacks += 1
            self.assertEqual(r.episode_timestep, self.num_callbacks * callback_timestep_frequency)

        runner.run(
            num_episodes=1, callback=callback,
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )

        self.is_callback1 = False
        self.is_callback2 = False

        def callback1(r):
            self.is_callback1 = True

        def callback2(r):
            self.is_callback2 = True

        runner.run(
            num_episodes=1, callback=[callback1, callback2],
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )
        runner.close()

        self.finished_test(assertion=(self.is_callback1 and self.is_callback2))

        # evaluation
        agent, environment = self.prepare()

        runner = Runner(agent=agent, environment=environment)

        self.num_evaluations = 0
        evaluation_frequency = 3
        num_evaluation_iterations = 2

        def evaluation_callback(r):
            self.num_evaluations += 1
            self.assertEqual(r.episodes, self.num_evaluations * evaluation_frequency)
            self.assertEqual(len(r.evaluation_timesteps), num_evaluation_iterations)

        runner.run(
            num_episodes=6, use_tqdm=False, evaluation_callback=evaluation_callback,
            evaluation_frequency=evaluation_frequency,
            num_evaluation_iterations=num_evaluation_iterations
        )
        runner.close()

        self.finished_test()

    def test_parallel_runner(self):
        self.start_tests(name='parallel-runner')

        agent, environment1 = self.prepare(
            update=dict(unit='episodes', batch_size=1), parallel_interactions=2
        )
        environment2 = copy.deepcopy(environment1)

        runner = ParallelRunner(agent=agent, environments=[environment1, environment2])
        runner.run(num_episodes=5, use_tqdm=False)
        runner.close()

        self.finished_test()

        # callback
        agent, environment1 = self.prepare(
            update=dict(unit='episodes', batch_size=1), parallel_interactions=2
        )
        environment2 = copy.deepcopy(environment1)

        runner = ParallelRunner(agent=agent, environments=[environment1, environment2])

        callback_episode_frequency = 2
        self.num_callbacks = 0

        def callback(r, parallel):
            self.num_callbacks += 1
            self.assertEqual(r.episodes, self.num_callbacks * callback_episode_frequency)

        runner.run(
            num_episodes=5, callback=callback,
            callback_episode_frequency=callback_episode_frequency, use_tqdm=False
        )

        time.sleep(1)

        callback_timestep_frequency = 3

        def callback(r, parallel):
            self.assertEqual(r.episode_timestep[parallel] % callback_timestep_frequency, 0)

        runner.run(
            num_episodes=1, callback=callback,
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )

        self.is_callback1 = False
        self.is_callback2 = False

        def callback1(r, parallel):
            self.is_callback1 = True

        def callback2(r, parallel):
            self.is_callback2 = True

        runner.run(
            num_episodes=1, callback=[callback1, callback2],
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )
        runner.close()

        self.finished_test(assertion=(self.is_callback1 and self.is_callback2))

        # evaluation
        agent, environment1 = self.prepare(
            update=dict(unit='episodes', batch_size=1), parallel_interactions=2
        )
        environment2 = copy.deepcopy(environment1)
        evaluation_environment = copy.deepcopy(environment1)

        runner = ParallelRunner(
            agent=agent, environments=[environment1, environment2],
            evaluation_environment=evaluation_environment
        )

        self.num_evaluations = 0

        def evaluation_callback(r):
            self.num_evaluations += 1

        runner.run(num_episodes=5, use_tqdm=False, evaluation_callback=evaluation_callback)
        runner.close()

        self.assertGreaterEqual(self.num_evaluations, 1)

        self.finished_test()
