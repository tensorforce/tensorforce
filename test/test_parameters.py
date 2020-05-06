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

import numpy as np

from tensorforce import util
from test.unittest_base import UnittestBase


class TestParameters(UnittestBase, unittest.TestCase):

    def float_unittest(self, exploration):
        agent, environment = self.prepare(min_timesteps=3, exploration=exploration)

        states = environment.reset()
        actions = agent.act(states=states)
        exploration1 = agent.model.exploration.value().numpy().item()
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        actions = agent.act(states=states)
        exploration2 = agent.model.exploration.value().numpy().item()
        if not isinstance(exploration, dict) or exploration['type'] == 'constant':
            self.assertEqual(exploration2, exploration1)
        else:
            self.assertNotEqual(exploration2, exploration1)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        self.finished_test()

    def int_unittest(self, horizon):
        agent, environment = self.prepare(min_timesteps=3, reward_estimation=dict(horizon=horizon))

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        horizon1 = agent.model.estimator.horizon.value().numpy().item()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        horizon2 = agent.model.estimator.horizon.value().numpy().item()
        if not isinstance(horizon, dict) or horizon['type'] == 'constant':
            self.assertEqual(horizon2, horizon1)
        else:
            self.assertNotEqual(horizon2, horizon1)

        agent.close()
        environment.close()

        self.finished_test()

    def test_constant(self):
        self.start_tests(name='constant')

        exploration = 0.1
        self.float_unittest(exploration=exploration)

        horizon = 4
        self.int_unittest(horizon=horizon)

    def test_decaying(self):
        # SPECIFICATION.MD
        self.start_tests(name='decaying')

        exploration = dict(
            type='decaying', unit='timesteps', decay='exponential', initial_value=0.1,
            decay_steps=1, decay_rate=0.5
        )
        self.float_unittest(exploration=exploration)

        horizon = dict(
            type='decaying', dtype='int', unit='timesteps', decay='polynomial',
            initial_value=2.0, decay_steps=2, final_value=4.0, power=1.0
        )
        self.int_unittest(horizon=horizon)

    def test_ornstein_uhlenbeck(self):
        self.start_tests(name='ornstein-uhlenbeck')

        exploration = dict(type='ornstein_uhlenbeck', absolute=True)
        self.float_unittest(exploration=exploration)

    def test_piecewise_constant(self):
        self.start_tests(name='piecewise-constant')

        exploration = dict(
            type='piecewise_constant', unit='timesteps', boundaries=[1], values=[0.1, 0.0]
        )
        self.float_unittest(exploration=exploration)

        horizon = dict(
            type='piecewise_constant', dtype='int', unit='timesteps', boundaries=[1], values=[1, 2]
        )
        self.int_unittest(horizon=horizon)

    def test_random(self):
        self.start_tests(name='random')

        exploration = dict(type='random', distribution='uniform')
        self.float_unittest(exploration=exploration)
