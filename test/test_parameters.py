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

import numpy as np

from tensorforce import util
from test.unittest_base import UnittestBase


class TestParameters(UnittestBase, unittest.TestCase):

    require_observe = True

    def float_unittest(self, exploration):
        agent, environment = self.prepare(min_timesteps=3, exploration=exploration)

        states = environment.reset()
        actions, exploration_output1 = agent.act(states=states, query='exploration')
        self.assertIsInstance(exploration_output1, util.np_dtype(dtype='float'))
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        if not isinstance(exploration, dict) or exploration['type'] == 'constant':
            actions, exploration_output2 = agent.act(states=states, query='exploration')
            self.assertEqual(exploration_output2, exploration_output1)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

        else:
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

        self.finished_test()

    def long_unittest(self, horizon):
        agent, environment = self.prepare(
            min_timesteps=3, reward_estimation=dict(horizon=horizon), memory=20
        )

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        _, horizon_output1 = agent.observe(terminal=terminal, reward=reward, query='horizon')
        self.assertIsInstance(horizon_output1, util.np_dtype(dtype='long'))

        if not isinstance(horizon, dict) or horizon['type'] == 'constant':
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            _, horizon_output2 = agent.observe(terminal=terminal, reward=reward, query='horizon')
            self.assertEqual(horizon_output2, horizon_output1)

        else:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            _, horizon_output2 = agent.observe(terminal=terminal, reward=reward, query='horizon')
            self.assertNotEqual(horizon_output2, horizon_output1)

        actions = agent.act(states=states)
        _, terminal, reward = environment.execute(actions=actions)
        horizon_input = 3
        _, horizon_output = agent.observe(
            terminal=terminal, reward=reward, query='horizon',
            **{'estimator/horizon': horizon_input}
        )
        self.assertEqual(
            horizon_output, np.asarray(horizon_input, dtype=util.np_dtype(dtype='long'))
        )

        agent.close()
        environment.close()

        self.finished_test()

    def test_constant(self):
        self.start_tests(name='constant')

        exploration = dict(type='constant', value=0.1)
        self.float_unittest(exploration=exploration)

        horizon = dict(type='constant', value=1, dtype='long')
        self.long_unittest(horizon=horizon)

        exploration = 0.1
        self.float_unittest(exploration=exploration)

        horizon = 1
        self.long_unittest(horizon=horizon)

    def test_decaying(self):
        # SPECIFICATION.MD
        self.start_tests(name='decaying')

        exploration = dict(
            type='decaying', unit='timesteps', decay='exponential', initial_value=0.1,
            decay_steps=1, decay_rate=0.5
        )
        self.float_unittest(exploration=exploration)

        horizon = dict(
            type='decaying', dtype='long', unit='timesteps', decay='polynomial',
            initial_value=2.0, decay_steps=2, final_value=4.0, power=1.0
        )
        self.long_unittest(horizon=horizon)

    def test_ornstein_uhlenbeck(self):
        self.start_tests(name='ornstein-uhlenbeck')

        exploration = dict(type='ornstein_uhlenbeck')
        self.float_unittest(exploration=exploration)

    def test_piecewise_constant(self):
        self.start_tests(name='piecewise-constant')

        # first act at timestep 0
        exploration = dict(
            type='piecewise_constant', unit='timesteps', boundaries=[0], values=[0.1, 0.0]
        )
        self.float_unittest(exploration=exploration)

        # first observe at timestep 1
        horizon = dict(
            type='piecewise_constant', dtype='long', unit='timesteps', boundaries=[1],
            values=[1, 2]
        )
        self.long_unittest(horizon=horizon)

    def test_random(self):
        self.start_tests(name='random')

        exploration = dict(type='random', distribution='normal')
        self.float_unittest(exploration=exploration)
