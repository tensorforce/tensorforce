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

from tensorforce import Configuration
from tensorforce.agents import RandomAgent
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner
from tensorforce.tests import reward_threshold


class TestRandomAgent(unittest.TestCase):

    def test_discrete(self):
        environment = MinimalTest(definition=False)
        config = Configuration(
            states=environment.states,
            actions=environment.actions
        )
        agent = RandomAgent(config=config)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x / l >= 0.9 for x, l in zip(r.episode_rewards[-100:], r.episode_lengths[-100:]))

        runner.run(episodes=1000, episode_finished=episode_finished)
        print('Random agent (discrete): ' + str(runner.episode))
        self.assertTrue(runner.episode == 1000)

    def test_continuous(self):
        environment = MinimalTest(definition=True)
        config = Configuration(
            states=environment.states,
            actions=environment.actions
        )
        agent = RandomAgent(config=config)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-100:],
                                                                                        r.episode_lengths[-100:]))

        runner.run(episodes=1000, episode_finished=episode_finished)
        print('Random agent (continuous): ' + str(runner.episode))
        self.assertTrue(runner.episode == 1000)

    def test_multi(self):
        environment = MinimalTest(definition=[False, (False, 2), (False, (1, 2)), (True, (1, 2))])
        config = Configuration(
            states=environment.states,
            actions=environment.actions
        )
        agent = RandomAgent(config=config)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 20 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-20:],
                                                                                       r.episode_lengths[-20:]))

        runner.run(episodes=1000, episode_finished=episode_finished)
        print('Random agent (multi-state/action): ' + str(runner.episode))
        self.assertTrue(runner.episode == 1000)
