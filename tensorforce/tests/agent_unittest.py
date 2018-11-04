# Copyright 2018 TensorForce Team. All Rights Reserved.
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

from tensorforce.tests.unittest_base import UnittestBase
from tensorforce.tests.unittest_environment import UnittestEnvironment


class AgentUnittest(UnittestBase):
    """
    Collection of unit-tests for agent functionality.
    """

    config = None

    # Flags for exclusion of action types.
    exclude_bool_action = False
    exclude_int_action = False
    exclude_float_action = False
    exclude_bounded_action = False
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
        environment = UnittestEnvironment(states=states, actions=actions)

        network = [
            dict(type='embedding', num_embeddings=2, size=32),
            dict(type='dense', size=32), dict(type='dense', size=32)
        ]

        config = self.__class__.config

        self.unittest(name='bool', environment=environment, network=network, config=config)

    def test_int(self):
        """
        Unit-test for integer state and action.
        """
        states = dict(type='int', shape=(), num_states=3)
        if self.__class__.exclude_int_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='int', shape=(), num_actions=3)
        environment = UnittestEnvironment(states=states, actions=actions)

        network = [
            dict(type='embedding', num_embeddings=3, size=32),
            dict(type='dense', size=32), dict(type='dense', size=32)
        ]

        config = self.__class__.config

        self.unittest(name='int', environment=environment, network=network, config=config)

    def test_float(self):
        """
        Unit-test for float state and action.
        """
        states = dict(type='float', shape=(1,))
        if self.__class__.exclude_float_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='float', shape=())
        environment = UnittestEnvironment(states=states, actions=actions)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = self.__class__.config

        self.unittest(name='float', environment=environment, network=network, config=config)

    def test_bounded(self):
        """
        Unit-test for bounded float state and action.
        """
        states = dict(type='float', shape=(1,), min_value=-1.0, max_value=1.0)
        if self.__class__.exclude_bounded_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='float', shape=(), min_value=-1.0, max_value=1.0)
        environment = UnittestEnvironment(states=states, actions=actions)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = self.__class__.config

        self.unittest(name='bounded', environment=environment, network=network, config=config)

    def test_full(self):
        """
        Unit-test for all types of states and actions.
        """
        states = dict(
            bool=dict(type='bool', shape=(1,)),
            int=dict(type='int', shape=(2,), num_states=4),
            float=dict(type='float', shape=(1, 1, 2)),
            bounded=dict(type='float', shape=(), min_value=-0.5, max_value=0.5)
        )
        actions = dict()
        if not self.__class__.exclude_bool_action:
            actions['bool'] = dict(type='bool', shape=(1,))
        if not self.__class__.exclude_int_action:
            actions['int'] = dict(type='int', shape=(2,), num_actions=4)
        if not self.__class__.exclude_float_action:
            actions['float'] = dict(type='float', shape=(1, 1))
        if not self.__class__.exclude_bounded_action:
            actions['bounded'] = dict(type='float', shape=(2,), min_value=-0.5, max_value=0.5)
        environment = UnittestEnvironment(states=states, actions=actions)

        network = [
            [
                dict(type='input', names='bool'),
                dict(type='embedding', num_embeddings=2, size=16),
                dict(type='lstm', size=8),
                dict(type='output', name='bool-emb')
            ],
            [
                dict(type='input', names='int'),
                dict(type='embedding', num_embeddings=4, size=16),
                dict(type='lstm', size=8),
                dict(type='output', name='int-emb')
            ],
            [
                dict(type='input', names='float'),
                dict(type='conv2d', size=16),
                dict(type='global_pooling', pooling='max'),
                dict(type='output', name='float-emb')
            ],
            [
                dict(type='input', names='bounded'),
                dict(type='global_pooling', pooling='concat'),
                dict(type='dense', size=16),
                dict(type='output', name='bounded-emb')
            ],
            [
                dict(
                    type='input', names=['bool-emb', 'int-emb', 'float-emb', 'bounded-emb'],
                    aggregation_type='product'
                ),
                dict(type='dense', size=16)
            ]
        ]

        config = self.__class__.config

        self.unittest(name='full', environment=environment, network=network, config=config)
