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

import unittest
import sys

from tensorforce import util
from tensorforce.agents import VPGAgent
from tensorforce.tests.agent_unittest import UnittestBase


class TestParameters(UnittestBase, unittest.TestCase):

    agent = VPGAgent

    def parameter_unittest(self, name, exploration):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        agent, environment = self.prepare(
            name=name, states=states, actions=actions, network=network, exploration=exploration
        )

        agent.initialize()
        states = environment.reset()

        actions, exploration_output1 = agent.act(states=states, query='exploration')
        self.assertIsInstance(exploration_output1, util.np_dtype(dtype='float'))

        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        if name != 'constant':
            actions, exploration_output2 = agent.act(states=states, query='exploration')
            self.assertNotEqual(exploration_output2, exploration_output1)

            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

        exploration_input = 0.5
        actions, exploration_output = agent.act(
            states=states, query='exploration', exploration=exploration_input
        )
        self.assertEqual(exploration_output, exploration_input)

        agent.close()
        environment.close()
        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_constant(self):
        self.parameter_unittest(name='constant', exploration=0.1)

    def test_random(self):
        exploration = dict(type='random', distribution='uniform')
        self.parameter_unittest(name='random', exploration=exploration)

    def test_piecewise_constant(self):
        exploration = dict(
            type='piecewise_constant', dtype='float', unit='timesteps', boundaries=[0],
            values=[0.1, 0.0]
        )
        self.parameter_unittest(name='piecewise-constant', exploration=exploration)

    def test_decaying(self):
        exploration = dict(
            type='decaying', unit='timesteps', decay='exponential', initial_value=0.1,
            decay_steps=1, decay_rate=0.5
        )
        self.parameter_unittest(name='decaying', exploration=exploration)

    def test_ornstein_uhlenbeck(self):
        exploration = dict(type='ornstein_uhlenbeck')
        self.parameter_unittest(name='ornstein-uhlenbeck', exploration=exploration)
