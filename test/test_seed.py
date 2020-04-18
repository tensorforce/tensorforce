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

        print_environment = False
        print_agent = False

        states = environment.reset()
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([2, 3])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([-0.11054066, 1.02017271])
            ))

        actions = agent.act(states=states)
        if print_agent:
            print(actions['int_action'])
            print(actions['float_action'])
        else:
            self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([0, 0])))
            self.assertTrue(expr=np.allclose(
                a=actions['float_action'], b=np.asarray([0.79587996, -0.7411721])
            ))

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
            print(terminal, reward, updated)
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([2, 2])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([1.2565714, 0.2967472])
            ))
            self.assertFalse(expr=terminal)
            self.assertEqual(first=reward, second=0.6888437030500962)
            self.assertFalse(expr=updated)

        actions = agent.act(states=states)
        if print_agent:
            print(actions['int_action'])
            print(actions['float_action'])
        else:
            self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([1, 0])))
            self.assertTrue(expr=np.allclose(
                a=actions['float_action'],b=np.asarray([-0.58322495, -0.08754656])
            ))

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
            print(terminal, reward, updated)
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([0, 2])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([-0.13370156, 1.07774381])
            ))
            self.assertFalse(expr=terminal)
            self.assertEqual(first=reward, second=-0.15885683833831)
            self.assertFalse(expr=updated)

        actions = agent.act(states=states)
        if print_agent:
            print(actions['int_action'])
            print(actions['float_action'])
        else:
            self.assertTrue(expr=np.allclose(a=actions['int_action'], b=np.asarray([3, 1])))
            self.assertTrue(expr=np.allclose(
                a=actions['float_action'], b=np.asarray([0.33305427, -0.21438375])
            ))

        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        if print_environment:
            print(states['int_state'])
            print(states['float_state'])
            print(terminal, reward, updated)
        else:
            self.assertTrue(expr=np.allclose(a=states['int_state'], b=np.asarray([1, 3])))
            self.assertTrue(expr=np.allclose(
                a=states['float_state'], b=np.asarray([0.42808095, -1.03978785])
            ))
            self.assertFalse(expr=terminal)
            self.assertEqual(first=reward, second=0.02254944273721704)
            self.assertFalse(expr=updated)
