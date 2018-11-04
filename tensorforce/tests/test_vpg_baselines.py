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

import unittest

from tensorforce.agents import VPGAgent
from tensorforce.tests.unittest_base import UnittestBase
from tensorforce.tests.unittest_environment import UnittestEnvironment


class TestVPGBaselines(UnittestBase, unittest.TestCase):

    agent = VPGAgent

    def test_baseline_states(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='mlp', sizes=[32, 32]),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='baseline-states', environment=environment, network=network, config=config
        )

    def test_baseline_network(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='network',
            baseline=dict(type='mlp', sizes=[32, 32]),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='baseline-network', environment=environment, network=network, config=config
        )

    def test_baseline_no_optimizer(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='mlp', sizes=[32, 32]),
        )

        self.unittest(
            name='baseline-no-optimizer', environment=environment, network=network, config=config
        )

    def test_baseline_gae(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='mlp', sizes=[32, 32]),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3),
            gae_lambda=0.95
        )

        self.unittest(
            name='baseline-gae', environment=environment, network=network, config=config
        )

    def test_aggregate_baseline(self):
        states = dict(
            state1=dict(type='float', shape=(1,)),
            state2=dict(type='float', shape=(1, 1, 1)),
        )
        environment = UnittestEnvironment(states=states, actions=dict(type='float', shape=()))

        network = [
            [
                dict(type='input', names='state1'),
                dict(type='dense', size=16),
                dict(type='output', name='state1-emb')
            ],
            [
                dict(type='input', names='state2'),
                dict(type='conv2d', size=16),
                dict(type='global_pooling', pooling='max'),
                dict(type='output', name='state2-emb')
            ],
            [
                dict(type='input', names=['state1-emb', 'state2-emb'], aggregation_type='product'),
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
            name='aggregate-baseline', environment=environment, network=network, config=config
        )

    def test_cnn_baseline(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1, 1, 1)), actions=dict(type='float', shape=())
        )

        network = [
            dict(type='conv2d', size=32), dict(type='global_pooling', pooling='max'),
            dict(type='dense', size=32)
        ]

        config = dict(
            baseline_mode='states',
            baseline=dict(type='cnn', conv_sizes=[32], dense_sizes=[32]),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='cnn-baseline', environment=environment, network=network, config=config
        )

    def test_network_baseline(self):
        states = dict(
            bool=dict(type='bool', shape=(1,)),
            int=dict(type='int', shape=(2,), num_states=4),
            float=dict(type='float', shape=(1, 1, 2)),
            bounded=dict(type='float', shape=(), min_value=-0.5, max_value=0.5)
        )
        environment = UnittestEnvironment(states=states, actions=dict(type='float', shape=()))

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

        config = dict(
            baseline_mode='states',
            baseline=dict(type='network', network=network),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='network-baseline', environment=environment, network=network, config=config
        )
