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
from tensorforce.agents.ppo_agent import PPOAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestPPOAgent(unittest.TestCase):
    def test_discrete(self):
        passed = 0

        # TRPO can occasionally have numerical issues so we allow for 1 in 5 to fail on Travis
        for _ in xrange(5):
            environment = MinimalTest(continuous=False)
            config = Configuration(
                batch_size=20,
                cg_iterations=20,
                cg_damping=0.001,
                entropy_penalty=0.01,
                loss_clipping=0.1,
                epochs=10,
                optimizer_batch_size=10,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([
                    dict(type='dense', size=32, activation='tanh'),
                    dict(type='dense', size=32, activation='tanh')
                ])
            )
            agent = PPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=2000, episode_finished=episode_finished)
            print('PPO Agent (discrete): ' + str(runner.episode))

            if runner.episode < 2000:
                passed += 1

        print('PPO discrete agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)

    def test_continuous(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(continuous=True)
            config = Configuration(
                batch_size=32,
                learning_rate=0.001,
                entropy_penalty=0.001,
                generalized_advantage_estimation=True,
                normalize_advantage=False,
                gae_lambda=0.97,
                loss_clipping=0.2,
                epochs=5,
                optimizer_batch_size=8,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([
                    dict(type='dense', size=32, activation='tanh'),
                    dict(type='dense', size=32, activation='tanh')
                ])
            )
            agent = PPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=2000, episode_finished=episode_finished)
            print('PPO Agent (continuous): ' + str(runner.episode))

            if runner.episode < 2000:
                passed += 1

        print('PPO continuous agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)
