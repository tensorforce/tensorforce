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
from tensorforce.agents import CategoricalDQNAgent
from tensorforce.core.networks import layered_network_builder, layers
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestCategoricalDQNAgent(unittest.TestCase):

    def test_discrete(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(definition=False)
            config = Configuration(
                batch_size=8,
                learning_rate=0.001,
                memory_capacity=800,
                first_update=80,
                target_update_frequency=20,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder([
                    dict(type='dense', size=32),
                    dict(type='dense', size=32)
                ])
            )
            agent = CategoricalDQNAgent(config=config)
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=1000, episode_finished=episode_finished)
            print('Categorical DQN agent: ' + str(runner.episode))
            if runner.episode < 1000:
                passed += 1

        print('Categorical DQN agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)

    # def test_multi(self):
    #     passed = 0
    #
    #     def network_builder(inputs):
    #         layer = layers['dense']
    #         state0 = layer(x=layer(x=inputs['state0'], size=32), size=32)
    #         state1 = layer(x=layer(x=inputs['state1'], size=32), size=32)
    #         return state0 * state1
    #
    #     for _ in xrange(5):
    #         environment = MinimalTest(definition=[False, (False, 2)])
    #         config = Configuration(
    #             batch_size=8,
    #             learning_rate=0.001,
    #             memory_capacity=800,
    #             first_update=80,
    #             target_update_frequency=20,
    #             states=environment.states,
    #             actions=environment.actions,
    #             network=network_builder
    #         )
    #         agent = CategoricalDQNAgent(config=config)
    #         runner = Runner(agent=agent, environment=environment)
    #
    #         def episode_finished(r):
    #             return r.episode < 15 or not all(x >= 1.0 for x in r.episode_rewards[-15:])
    #
    #         runner.run(episodes=2000, episode_finished=episode_finished)
    #         print('Categorical DQN agent (multi-state/action): ' + str(runner.episode))
    #         if runner.episode < 2000:
    #             passed += 1
    #
    #     print('Categorical DQN agent (multi-state/action) passed = {}'.format(passed))
    #     self.assertTrue(passed >= 2)
