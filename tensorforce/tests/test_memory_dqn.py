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
from tensorforce.agents import DQNAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner
from tensorforce.tests import reward_threshold


class TestMemoryDQN(unittest.TestCase):

    def test_replay(self):
        environment = MinimalTest(definition=[(False, (1, 2))])
        config = Configuration(
            batch_size=8,
            learning_rate=0.001,
            memory_capacity=50,
            memory=dict(
                type='replay',
                random_sampling=True
            ),
            first_update=20,
            target_update_frequency=10,
            states=environment.states,
            actions=environment.actions,
            network=layered_network_builder([
                dict(type='dense', size=32),
                dict(type='dense', size=32)
            ])
        )
        agent = DQNAgent(config=config)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-100:], r.episode_lengths[-100:]))

        runner.run(episodes=1000, episode_finished=episode_finished)
        print('Replay memory DQN: ' + str(runner.episode))

    def test_prioritized_replay(self):
        environment = MinimalTest(definition=[(False, (1, 2))])
        config = Configuration(
            batch_size=8,
            learning_rate=0.001,
            memory_capacity=50,
            memory='prioritized_replay',
            first_update=20,
            target_update_frequency=10,
            states=environment.states,
            actions=environment.actions,
            network=layered_network_builder([
                dict(type='dense', size=32),
                dict(type='dense', size=32)
            ])
        )
        agent = DQNAgent(config=config)
        runner = Runner(agent=agent, environment=environment)

        def episode_finished(r):
            return r.episode < 100 or not all(x / l >= reward_threshold for x, l in zip(r.episode_rewards[-100:],
                                                                                        r.episode_lengths[-100:]))

        runner.run(episodes=1000, episode_finished=episode_finished)
        print('Prioritized replay memory DQN: ' + str(runner.episode))
