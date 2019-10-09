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

from test.unittest_base import UnittestBase
from tensorforce.agents import Agent
from tensorforce.environments import Environment


class TestSeed(UnittestBase, unittest.TestCase):

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

        agent, environment = self.prepare(states=states, actions=actions, seed=0)

        agent.initialize()

        states = environment.reset()
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([2, 1])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([-1.33221165, -1.96862469]))
        )

        actions = agent.act(states=states)
        self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([0, 3])))
        self.assertTrue(
            expr=np.allclose(a=actions['float_action'], b=np.asarray([0.6095624, 1.0851405]))
        )

        states, terminal, reward = environment.execute(actions=actions)
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([1, 3])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([0.03544277, -0.4779011]))
        )
        self.assertEqual(first=terminal, second=False)
        self.assertEqual(first=reward, second=0.515908805880605)

        actions = agent.act(states=states)
        self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([3, 3])))
        self.assertTrue(
            expr=np.allclose(a=actions['float_action'], b=np.asarray([-0.3195898, 0.28039014]))
        )

        states, terminal, reward = environment.execute(actions=actions)
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([1, 0])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([0.21348005, -1.20857365]))
        )
        self.assertEqual(first=terminal, second=False)
        self.assertEqual(first=reward, second=-0.15885683833831)

        actions = agent.act(states=states)
        self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([2, 0])))
        self.assertTrue(
            expr=np.allclose(a=actions['float_action'], b=np.asarray([1.6682274, 0.34829643]))
        )

        states, terminal, reward = environment.execute(actions=actions)
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([1, 1])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([0.01793846, -2.28547991]))
        )
        self.assertEqual(first=terminal, second=False)
        self.assertEqual(first=reward, second=-0.4821664994140733)

        actions = agent.act(states=states)
        self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([1, 2])))
        self.assertTrue(
            expr=np.allclose(a=actions['float_action'], b=np.asarray([1.4179794, 0.5713567]))
        )

        states, terminal, reward = environment.execute(actions=actions)
        self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([3, 3])))
        self.assertTrue(
            expr=np.allclose(a=states['float_state'], b=np.asarray([0.0086279 , 0.52700421]))
        )
        self.assertEqual(first=terminal, second=True)
        self.assertEqual(first=reward, second=0.02254944273721704)
