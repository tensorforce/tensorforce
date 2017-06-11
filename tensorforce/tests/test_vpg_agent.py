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
from tensorforce.agents import VPGAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestVPGAgent(unittest.TestCase):

    def test_discrete(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(continuous=False)
            config = Configuration(
                batch_size=8,
                learning_rate=0.001,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([dict(type='dense', size=32)])
            )
            agent = VPGAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=10000, episode_finished=episode_finished)
            print('VPG Agent (discrete): ' + str(runner.episode))

            if runner.episode < 10000:
                passed += 1
                print('passed')
            else:
                print('failed')

        print('VPG discrete agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)

    def test_continuous(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(continuous=True)
            config = Configuration(
                batch_size=8,
                learning_rate=0.001,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([dict(type='dense', size=32)])
            )
            agent = VPGAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=5000, episode_finished=episode_finished)
            print('VPG Agent (continuous): ' + str(runner.episode))
            if runner.episode < 5000:
                passed += 1
                print('passed')
            else:
                print('failed')

        print('VPG continuous agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)