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

from tensorforce.agents import VPGAgent
from tensorforce.tests.unittest_base import UnittestBase


class TestBaselines(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))

    def test_baseline_states(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='mlp', sizes=[32, 32]),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='baseline-states', states=states, actions=actions, network=network, **config
        )

    def test_baseline_network(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='network',
            baseline=dict(type='mlp', sizes=[32, 32]),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='baseline-network', states=states, actions=actions, network=network, **config
        )

    def test_baseline_no_optimizer(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='mlp', sizes=[32, 32]),
        )

        self.unittest(
            name='baseline-no-optimizer', states=states, actions=actions, network=network, **config
        )

    def test_baseline_gae(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='mlp', sizes=[32, 32]),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3),
            gae_lambda=0.95
        )

        self.unittest(
            name='baseline-gae', states=states, actions=actions, network=network, **config
        )

    def test_aggregated_baseline(self):
        states = dict(
            state1=dict(type='float', shape=(1,)),
            state2=dict(type='float', shape=(1, 1, 2))
        )

        actions = dict(type='int', shape=(), num_values=3)

        network = [
            [
                dict(type='retrieve', tensors='state1'),
                dict(type='dense', size=16),
                dict(type='register', tensor='state1-emb')
            ],
            [
                dict(type='retrieve', tensors='state2'),
                dict(type='conv2d', size=16),
                dict(type='pooling', reduction='max'),
                dict(type='register', tensor='state2-emb')
            ],
            [
                dict(type='retrieve', tensors=('state1-emb', 'state2-emb'), aggregation='product'),
                dict(type='dense', size=16)
            ]
        ]

        config = dict(
            baseline_mode='states',
            baseline=dict(
                type='aggregated',
                baselines=dict(
                    state1=dict(type='mlp', sizes=[16, 16]),
                    state2=dict(type='cnn', conv_sizes=[16], dense_sizes=[16])
                )
            ),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='aggregated-baseline', states=states, actions=actions, network=network, **config
        )

    def test_network_baseline(self):
        states = dict(
            bool_state=dict(type='bool', shape=(1,)),
            int_state=dict(type='int', shape=(2,), num_values=4),
            float_state=dict(type='float', shape=(1, 1, 2)),
            bounded_state=dict(type='float', shape=(), min_value=-0.5, max_value=0.5)
        )

        actions = dict(type='int', shape=(), num_values=3)

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
                dict(type='lstm', size=16),
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
                dict(type='dense', size=16)  # internal_lstm not yet supported!!!!!!!!!!!
            ]
        ]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='network', network=network),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='network-baseline', states=states, actions=actions, network=network, **config
        )
