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

from test.unittest_base import UnittestBase


class TestSeed(UnittestBase, unittest.TestCase):

    def test_seed(self):
        self.start_tests()

        states = dict(
            int_state=dict(type='int', shape=(2,), num_values=4),
            float_state=dict(type='float', shape=(2,), min_value=1.0, max_value=2.0),
        )
        actions = dict(
            int_action=dict(type='int', shape=(2,), num_values=4),
            float_action=dict(type='float', shape=(2,), min_value=1.0, max_value=2.0),
        )

        agent, environment = self.prepare(
            states=states, actions=actions, exploration=0.5,
            config=dict(seed=0, eager_mode=True, create_debug_assertions=True)
        )

        print_environment = False
        print_agent = False

        states = environment.reset()
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([3, 1])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([-0.72559738, -1.38336396])
            ))

        actions = agent.act(states=states)
        if print_agent:
            print(actions['int_action'])
            print(actions['float_action'])
        else:
            self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([2, 2])))
            self.assertTrue(expr=np.allclose(
                a=actions['float_action'], b=np.asarray([-0.5055479, -0.35684115])
            ))

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
            print(terminal, reward, updated)
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([0, 1])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([-1.15477553, 0.94183434])
            ))
            self.assertFalse(expr=terminal)
            self.assertEqual(first=reward, second=0.6888437030500962)
            self.assertFalse(expr=updated)

        actions = agent.act(states=states)
        if print_agent:
            print(actions['int_action'])
            print(actions['float_action'])
        else:
            self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([0, 3])))
            self.assertTrue(expr=np.allclose(
                a=actions['float_action'], b=np.asarray([-1.0440513, 0.24466835])
            ))

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
            print(terminal, reward, updated)
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([3, 1])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([-0.62808756, -0.48102712])
            ))
            self.assertFalse(expr=terminal)
            self.assertEqual(first=reward, second=0.515908805880605)
            self.assertFalse(expr=updated)

        actions = agent.act(states=states)
        if print_agent:
            print(actions['int_action'])
            print(actions['float_action'])
        else:
            self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([1, 2])))
            self.assertTrue(expr=np.allclose(
                a=actions['float_action'], b=np.asarray([-0.53231376, -0.09416733])
            ))

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
            print(terminal, reward, updated)
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([2, 0])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([-0.11757479, 0.37312016])
            ))
            self.assertFalse(expr=terminal)
            self.assertEqual(first=reward, second=-0.15885683833831)
            self.assertFalse(expr=updated)
