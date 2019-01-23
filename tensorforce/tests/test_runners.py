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
import sys
import time
import unittest

from tensorforce.agents import VPGAgent
from tensorforce.execution import ParallelRunner, Runner
from tensorforce.tests.unittest_base import UnittestBase


class TestRunners(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))

    def test_runner_callback(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        agent, environment = self.prepare(
            name='runner-callback', states=states, actions=actions, network=network
        )
        environment.timestep_range = (6, 10)

        runner = Runner(agent=agent, environment=environment)

        callback_episode_frequency = 2
        self.num_callbacks = 0

        def callback(r):
            self.num_callbacks += 1
            self.assertEqual(r.episode, self.num_callbacks * callback_episode_frequency)

        runner.run(
            num_episodes=10, callback=callback,
            callback_episode_frequency=callback_episode_frequency
        )

        callback_timestep_frequency = 3
        self.num_callbacks = 0

        def callback(r):
            self.num_callbacks += 1
            self.assertEqual(r.episode_timestep, self.num_callbacks * callback_timestep_frequency)

        runner.run(
            num_episodes=11, callback=callback,
            callback_timestep_frequency=callback_timestep_frequency
        )

        runner.close()
        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_runner_evaluation(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        agent, environment = self.prepare(
            name='runner-evaluation', states=states, actions=actions, network=network
        )

        runner = Runner(agent=agent, environment=environment)

        self.num_evaluations = 0
        evaluation_frequency = 3
        max_evaluation_timesteps = 2
        num_evaluation_iterations = 2

        def evaluation_callback(r):
            self.num_evaluations += 1
            self.assertEqual(r.episode, self.num_evaluations * evaluation_frequency)
            self.assertEqual(len(r.evaluation_timesteps), num_evaluation_iterations)
            for num_timesteps in r.evaluation_timesteps:
                self.assertLessEqual(num_timesteps, max_evaluation_timesteps)

        runner.run(
            num_episodes=10, evaluation_callback=evaluation_callback,
            evaluation_frequency=evaluation_frequency,
            max_evaluation_timesteps=max_evaluation_timesteps,
            num_evaluation_iterations=num_evaluation_iterations
        )

        runner.close()
        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_parallel_runner(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        agent, environment1 = self.prepare(
            name='parallel-runner', states=states, actions=actions, network=network,
            parallel_interactions=2
        )
        environment2 = copy.deepcopy(environment1)

        runner = ParallelRunner(agent=agent, environments=[environment1, environment2])
        runner.run(num_episodes=10)
        runner.close()

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_parallel_runner_callback(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        agent, environment1 = self.prepare(
            name='parallel-runner-callback', states=states, actions=actions, network=network,
            parallel_interactions=2
        )
        environment1.timestep_range = (6, 10)
        environment2 = copy.deepcopy(environment1)

        runner = ParallelRunner(agent=agent, environments=[environment1, environment2])

        callback_episode_frequency = 2
        self.num_callbacks = 0

        def callback(r, parallel):
            self.num_callbacks += 1
            self.assertEqual(r.episode, self.num_callbacks * callback_episode_frequency)

        runner.run(
            num_episodes=10, callback=callback,
            callback_episode_frequency=callback_episode_frequency
        )

        time.sleep(1)

        callback_timestep_frequency = 3

        def callback(r, parallel):
            pass
            # self.assertEqual(r.episode_timestep[parallel] % callback_timestep_frequency, 0)

        runner.run(
            num_episodes=11, callback=callback,
            callback_timestep_frequency=callback_timestep_frequency
        )

        runner.close()
        sys.stdout.flush()
        self.assertTrue(expr=True)
