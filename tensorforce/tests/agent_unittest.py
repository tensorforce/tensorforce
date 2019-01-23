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

import sys

from tensorforce.tests.unittest_base import UnittestBase


class AgentUnittest(UnittestBase):
    """
    Collection of unit-tests for agent functionality.
    """

    # Flags for exclusion of action types.
    exclude_bool_action = False
    exclude_int_action = False
    exclude_float_action = False
    exclude_bounded_action = False
    exclude_lstm = False  # TEMP
    replacement_action = 'bool'

    def test_bool(self):
        """
        Unit-test for boolean state and action.
        """
        states = dict(type='bool', shape=())

        if self.__class__.exclude_bool_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='bool', shape=())

        network = [
            dict(type='embedding', size=32),
            dict(type='dense', size=32), dict(type='dense', size=32)
        ]

        self.unittest(name='bool', states=states, actions=actions, network=network)

    def test_int(self):
        """
        Unit-test for integer state and action.
        """
        states = dict(type='int', shape=(), num_values=3)

        if self.__class__.exclude_int_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='int', shape=(), num_values=3)

        network = [
            dict(type='embedding', size=32),
            dict(type='dense', size=32), dict(type='dense', size=32)
        ]

        self.unittest(name='int', states=states, actions=actions, network=network)

    def test_float(self):
        """
        Unit-test for float state and action.
        """
        states = dict(type='float', shape=(1,))

        if self.__class__.exclude_float_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='float', shape=())

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        self.unittest(name='float', states=states, actions=actions, network=network)

    def test_bounded(self):
        """
        Unit-test for bounded float state and action.
        """
        states = dict(type='float', shape=(1,), min_value=-1.0, max_value=1.0)

        if self.__class__.exclude_bounded_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='float', shape=(), min_value=-1.0, max_value=1.0)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        self.unittest(name='bounded', states=states, actions=actions, network=network)

    def test_full(self):
        """
        Unit-test for all types of states and actions.
        """
        states = dict(
            bool_state=dict(type='bool', shape=(1,)),
            int_state=dict(type='int', shape=(2,), num_values=4),
            float_state=dict(type='float', shape=(1, 1, 2)),
            bounded_state=dict(type='float', shape=(), min_value=-0.5, max_value=0.5)
        )

        actions = dict()
        if not self.__class__.exclude_bool_action:
            actions['bool_action'] = dict(type='bool', shape=(1,))
        if not self.__class__.exclude_int_action:
            actions['int_action'] = dict(type='int', shape=(2,), num_values=4)
        if not self.__class__.exclude_float_action:
            actions['float_action'] = dict(type='float', shape=(1, 1))
        if not self.__class__.exclude_bounded_action:
            actions['bounded_action'] = dict(
                type='float', shape=(2,), min_value=-0.5, max_value=0.5
            )

        # Same in test_baselines.py
        network = [
            [
                dict(type='retrieve', tensors='bool_state'),
                dict(type='embedding', size=16),
                dict(type='conv1d', size=16),
                dict(type='pooling', reduction='max'),
                dict(type='register', tensor='bool-emb')
            ],
            [
                dict(type='retrieve', tensors='int_state'),
                dict(type='embedding', size=16),
                (
                    dict(type='pooling', reduction='max') if self.__class__.exclude_lstm
                    else dict(type='lstm', size=16)
                ),
                dict(type='register', tensor='int-emb')
            ],
            [
                dict(type='retrieve', tensors='float_state'),
                dict(type='conv2d', size=16),
                dict(type='pooling', reduction='max'),
                dict(type='register', tensor='float-emb')
            ],
            [
                dict(type='retrieve', tensors='bounded_state'),
                dict(type='pooling', reduction='concat'),
                dict(type='dense', size=16),
                dict(type='register', tensor='bounded-emb')
            ],
            [
                dict(
                    type='retrieve', tensors=('bool-emb', 'int-emb', 'float-emb', 'bounded-emb'),
                    aggregation='product'
                ),
                dict(type='internal_lstm', size=16)
            ]
        ]

        self.unittest(name='full', states=states, actions=actions, network=network)

    def test_query(self):
        """
        Unit-test for all types of states and actions.
        """
        states = dict(type='float', shape=(1,))

        actions = dict(type=self.__class__.replacement_action, shape=())

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        act_query = ('state', 'action', 'deterministic', 'update', 'timestep', 'episode')
        observe_query = (
            'state', 'action', 'reward', 'terminal', 'deterministic', 'update', 'timestep',
            'episode'
        )

        agent, environment = self.prepare(
            name='query', states=states, actions=actions, network=network, buffer_observe=False
        )

        agent.initialize()
        states = environment.reset()

        actions, query = agent.act(states=states, query=act_query)
        self.assertEqual(first=len(query), second=6)

        states, terminal, reward = environment.execute(actions=actions)

        query = agent.observe(terminal=terminal, reward=reward, query=observe_query)
        self.assertEqual(first=len(query), second=8)

        agent.close()
        environment.close()
        sys.stdout.flush()
        self.assertTrue(expr=True)
