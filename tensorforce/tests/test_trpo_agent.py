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
from tensorforce.core.networks import layered_network_builder, layers
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner
from tensorforce.tests import reward_threshold


class TestTRPOAgent(unittest.TestCase):

    def test_discrete(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(definition=False)
            config = Configuration(
                batch_size=8,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([
                    dict(type='dense', size=32, activation='tanh'),
                    dict(type='dense', size=32, activation='tanh')
                ])
            )
            agent = TRPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-100:], r.episode_lengths[-100:]))

            runner.run(episodes=1000, episode_finished=episode_finished)
            print('TRPO agent (discrete): ' + str(runner.episode))

            if runner.episode < 1000:
                passed += 1

        print('TRPO agent (discrete) passed = {}'.format(passed))
        self.assertTrue(passed >= 4)

    def test_continuous(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(definition=True)
            config = Configuration(
                batch_size=8,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([
                    dict(type='dense', size=32, activation='tanh'),
                    dict(type='dense', size=32, activation='tanh')
                ])
            )
            agent = TRPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-100:],
                                                                                            r.episode_lengths[-100:]))

            runner.run(episodes=1000, episode_finished=episode_finished)
            print('TRPO agent (continuous): ' + str(runner.episode))

            if runner.episode < 1000:
                passed += 1

        print('TRPO agent (continuous) passed = {}'.format(passed))
        self.assertTrue(passed >= 4)

    def test_multi(self):
        passed = 0

        def network_builder(inputs, **kwargs):
            layer = layers['dense']
            state0 = layer(x=layer(x=inputs['state0'], size=32, scope='state0-1'), size=32, scope='state0-2')
            state1 = layer(x=layer(x=inputs['state1'], size=32, scope='state1-1'), size=32, scope='state1-2')
            state2 = layer(x=layer(x=inputs['state2'], size=32, scope='state2-1'), size=32, scope='state2-2')
            return state0 * state1 * state2

        for _ in xrange(5):
            environment = MinimalTest(definition=[False, (False, 2), (True, 2)])
            config = Configuration(
                batch_size=8,
                states=environment.states,
                actions=environment.actions,
                network=network_builder
            )
            agent = TRPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 15 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-15:],
                                                                                           r.episode_lengths[-15:]))

            runner.run(episodes=2000, episode_finished=episode_finished)
            print('TRPO agent (multi-state/action): ' + str(runner.episode))
            if runner.episode < 2000:
                passed += 1

        print('TRPO agent (multi-state/action) passed = {}'.format(passed))
        self.assertTrue(passed >= 2)

    def test_beta(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(definition=True)
            actions = environment.actions
            actions['min_value'] = -0.5
            actions['max_value'] = 1.5

            config = Configuration(
                batch_size=8,
                max_kl_divergence=0.05,
                states=environment.states,
                actions=actions,
                network=layered_network_builder([
                    dict(type='dense', size=32, activation='tanh'),
                    dict(type='dense', size=32, activation='tanh')
                ])
            )
            agent = TRPOAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-100:],
                                                                                            r.episode_lengths[-100:]))

            runner.run(episodes=5000, episode_finished=episode_finished)
            print('TRPO agent (beta): ' + str(runner.episode))

            if runner.episode < 5000:
                passed += 1

        print('TRPO agent (beta) passed = {}'.format(passed))
        self.assertTrue(passed >= 4)
