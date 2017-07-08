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

from six.moves import xrange

import unittest

from tensorforce import Configuration
from tensorforce.agents import DQFDAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.execution import Runner


class TestDQFDAgent(unittest.TestCase):

    def test_dqfd_agent(self):
        passed = 0

        for _ in xrange(5):
            environment = MinimalTest(continuous=False)
            config = Configuration(
                batch_size=8,
                learning_rate=0.001,
                memory_capacity=800,
                first_update=80,
                repeat_update=4,
                target_update_frequency=20,
                demo_memory_capacity=100,
                demo_sampling_ratio=0.1,
                states=environment.states,
                actions=environment.actions,
                network=layered_network_builder(layers_config=[dict(type='dense', size=32, l2_regularization=0.0001)])
            )
            agent = DQFDAgent(config=config)

            # First generate demonstration data and pretrain
            demonstrations = list()
            terminal = True

            for n in xrange(50):
                if terminal:
                    state = environment.reset()
                action = 1
                state, reward, terminal = environment.execute(action=action)
                demonstration = dict(state=state, action=action, reward=reward, terminal=terminal, internal=[])
                demonstrations.append(demonstration)

            agent.import_demonstrations(demonstrations)
            agent.pretrain(steps=1000)

            # Normal training
            runner = Runner(agent=agent, environment=environment)

            def episode_finished(r):
                return r.episode < 100 or not all(x >= 1.0 for x in r.episode_rewards[-100:])

            runner.run(episodes=1000, episode_finished=episode_finished)
            print('DQFD Agent: ' + str(runner.episode))
            if runner.episode < 1000:
                passed += 1

        print('DQFD Agent passed = {}'.format(passed))
        self.assertTrue(passed >= 4)
