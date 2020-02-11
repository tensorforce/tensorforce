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

from tensorforce import Agent, Environment
from test.unittest_base import UnittestBase


class TestSeed(UnittestBase, unittest.TestCase):

    require_observe = True

    def test_seed(self):
        self.start_tests()

        states = dict(
            int_state=dict(type='int', shape=(2,), num_values=4),
            float_state=dict(type='float', shape=(2,)),
        )
        actions = dict(
            int_action=dict(type='int', shape=(2,), num_values=4),
            float_action=dict(type='float', shape=(2,)),
        )

        agent, environment = self.prepare(states=states, actions=actions, exploration=0.5, seed=0)

        states = environment.reset()
        # print(states['int_state'])
        # print(states['float_state'])
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([3, 2])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([1.95591231, 0.39009332]))
        )

        actions = agent.act(states=states)
        # print(actions['int_action'])
        # print(actions['float_action'])
        self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([3, 1])))
        self.assertTrue(
            expr=np.allclose(a=actions['float_action'], b=np.asarray([0.34490278, -0.30759263]))
        )

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        # print(states['int_state'])
        # print(states['float_state'])
        # print(terminal, reward, updated)
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([0, 2])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([-1.29363428, 0.67702681]))
        )
        self.assertFalse(expr=terminal)
        self.assertEqual(first=reward, second=0.515908805880605)
        self.assertFalse(expr=updated)

        actions = agent.act(states=states)
        # print(actions['int_action'])
        # print(actions['float_action'])
        self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([2, 3])))
        self.assertTrue(
            expr=np.allclose(a=actions['float_action'], b=np.asarray([-0.6053257, 0.29353103]))
        )

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        # print(states['int_state'])
        # print(states['float_state'])
        # print(terminal, reward, updated)
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([1, 0])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([0.66638308, -0.46071979]))
        )
        self.assertFalse(expr=terminal)
        self.assertEqual(first=reward, second=-0.4821664994140733)
        self.assertFalse(expr=updated)

        actions = agent.act(states=states)
        # print(actions['int_action'])
        # print(actions['float_action'])
        self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([2, 1])))
        self.assertTrue(
            expr=np.allclose(a=actions['float_action'], b=np.asarray([-0.11170869, -0.51122946]))
        )

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        # print(states['int_state'])
        # print(states['float_state'])
        # print(terminal, reward, updated)
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([1, 1])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([-0.71952667, -0.11861612]))
        )
        self.assertFalse(expr=terminal)
        self.assertEqual(first=reward, second=-0.19013172509917142)
        self.assertFalse(expr=updated)
