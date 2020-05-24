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

import unittest

from tensorforce import Runner
from test.unittest_base import UnittestBase


class TestRunner(UnittestBase, unittest.TestCase):

    def test_single(self):
        self.start_tests(name='single')

        agent = self.agent_spec()
        environment = self.environment_spec()
        runner = Runner(agent=agent, environment=environment)

        # default
        runner.run(num_episodes=3, use_tqdm=False)
        self.finished_test()

        # evaluation
        runner.run(num_episodes=1, use_tqdm=False, evaluation=False)
        self.finished_test()

        # episode callback
        callback_episode_frequency = 2
        self.num_callbacks = 0

        def callback(r, p):
            self.num_callbacks += 1
            self.assertEqual(r.episodes, self.num_callbacks * callback_episode_frequency)

        runner.run(
            num_episodes=5, callback=callback,
            callback_episode_frequency=callback_episode_frequency, use_tqdm=False
        )
        self.finished_test()

        # timestep callback
        callback_timestep_frequency = 3
        self.num_callbacks = 0

        def callback(r, p):
            self.num_callbacks += 1
            self.assertEqual(
                r.episode_timestep[p], self.num_callbacks * callback_timestep_frequency
            )

        runner.run(
            num_episodes=1, callback=callback,
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )
        self.finished_test()

        # multiple callbacks
        self.is_callback1 = False
        self.is_callback2 = False

        def callback1(r, p):
            self.is_callback1 = True

        def callback2(r, p):
            self.is_callback2 = True

        runner.run(
            num_episodes=1, callback=[callback1, callback2],
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )
        runner.close()
        self.finished_test(assertion=(self.is_callback1 and self.is_callback2))

    def test_unbatched(self):
        self.start_tests(name='unbatched')

        agent = self.agent_spec()
        environment = self.environment_spec()

        # default
        runner = Runner(agent=agent, environment=environment, num_parallel=2)
        runner.run(num_episodes=3, use_tqdm=False)
        runner.close()
        self.finished_test()

        # episode callback
        runner = Runner(agent=agent, environments=[environment, environment])
        callback_episode_frequency = 2
        self.num_callbacks = 0

        def callback(r, p):
            self.num_callbacks += 1
            if self.num_callbacks % 2 == 0:
                self.assertEqual(min(r.episode_timestep), 0)
            self.assertEqual(r.episodes, self.num_callbacks * callback_episode_frequency)

        runner.run(
            num_episodes=5, callback=callback,
            callback_episode_frequency=callback_episode_frequency, use_tqdm=False,
            sync_episodes=True
        )
        self.finished_test()

        # timestep callback
        callback_timestep_frequency = 3

        def callback(r, p):
            self.assertEqual(r.episode_timestep[p] % callback_timestep_frequency, 0)

        runner.run(
            num_episodes=2, callback=callback,
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )
        runner.close()
        self.finished_test()

        # evaluation synced
        runner = Runner(agent=agent, environment=environment, num_parallel=2, evaluation=True)
        self.num_evaluations = 0

        def evaluation_callback(r):
            self.num_evaluations += 1

        runner.run(
            num_episodes=1, use_tqdm=False, evaluation_callback=evaluation_callback,
            sync_episodes=True
        )
        self.finished_test(assertion=(self.num_evaluations == 1))

        # evaluation non-synced
        runner.run(num_episodes=1, use_tqdm=False, evaluation_callback=evaluation_callback)
        runner.close()
        self.finished_test(assertion=(self.num_evaluations >= 2))

    def test_batched(self):
        self.start_tests(name='batched')

        agent = self.agent_spec()
        environment = self.environment_spec()

        # default
        runner = Runner(agent=agent, environment=environment, num_parallel=2)
        runner.run(num_episodes=3, use_tqdm=False, batch_agent_calls=True)
        runner.close()
        self.finished_test()

        # episode callback
        runner = Runner(agent=agent, environments=[environment, environment])
        callback_episode_frequency = 2
        self.num_callbacks = 0

        def callback(r, p):
            self.num_callbacks += 1
            if self.num_callbacks % 2 == 0:
                self.assertEqual(min(r.episode_timestep), 0)
            self.assertEqual(r.episodes, self.num_callbacks * callback_episode_frequency)

        runner.run(
            num_episodes=5, callback=callback,
            callback_episode_frequency=callback_episode_frequency, use_tqdm=False,
            batch_agent_calls=True, sync_episodes=True
        )
        self.finished_test()

        # timestep callback
        callback_timestep_frequency = 3

        def callback(r, p):
            self.assertEqual(r.episode_timestep[p] % callback_timestep_frequency, 0)

        runner.run(
            num_episodes=2, callback=callback,
            callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False,
            batch_agent_calls=True
        )
        runner.close()
        self.finished_test()

        # evaluation synced
        runner = Runner(agent=agent, environment=environment, num_parallel=2, evaluation=True)
        self.num_evaluations = 0

        def evaluation_callback(r):
            self.num_evaluations += 1

        runner.run(
            num_episodes=1, use_tqdm=False, evaluation_callback=evaluation_callback,
            batch_agent_calls=True, sync_episodes=True
        )
        self.finished_test(assertion=(self.num_evaluations == 1))

        # evaluation non-synced
        runner.run(
            num_episodes=1, use_tqdm=False, evaluation_callback=evaluation_callback,
            batch_agent_calls=True
        )
        runner.close()
        self.finished_test(assertion=(self.num_evaluations >= 2))
