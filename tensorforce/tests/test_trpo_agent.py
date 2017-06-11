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
from six.moves import xrange

from tensorforce import Configuration
from tensorforce.agents import TRPOAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestTRPOAgent(unittest.TestCase):

    def test_discrete(self):
        passed = 0

        # TRPO can occasionally have numerical issues so we allow for 1 in 5 to fail on Travis
        for _ in xrange(5):
            environment = MinimalTest(continuous=False)
            config = Configuration(
                batch_size=8,
                learning_rate=0.0001,
                cg_iterations=20,
                cg_damping=0.001,
                line_search_steps=20,
                max_kl_divergence=0.05,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([dict(type='dense', size=32)])
            )
            agent = TRPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=10000, episode_finished=episode_finished)
            print('TRPO Agent (discrete): ' + str(runner.episode))

            if runner.episode < 10000:
                passed += 1
                print('passed')
            else:
                print('failed')

        print('TRPO discrete agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)

    def test_continuous(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(continuous=True)
            config = Configuration(
                batch_size=8,
                cg_iterations=20,
                cg_damping=0.001,
                line_search_steps=20,
                max_kl_divergence=0.05,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([dict(type='dense', size=32)])
            )
            agent = TRPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=10000, episode_finished=episode_finished)
            print('TRPO Agent (continuous): ' + str(runner.episode))

            if runner.episode < 10000:
                passed += 1
                print('passed')
            else:
                print('failed')

        print('TRPO continuous agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)