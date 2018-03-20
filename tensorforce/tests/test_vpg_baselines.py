# Copyright 2017 reinforce.io. All Rights Reserved.
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

from tensorforce.tests.base_test import BaseTest
from tensorforce.agents import VPGAgent
from tensorforce.core.networks import Dense, LayerBasedNetwork
from .minimal_test import MinimalTest


class TestVPGBaselines(BaseTest, unittest.TestCase):

    agent = VPGAgent

    def test_states_baseline(self):
        environment = MinimalTest(specification={'int': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        config = dict(
            update_mode=dict(
                unit='episodes',
                batch_size=4,
                frequency=4
            ),
            memory=dict(
                type='latest',
                include_next_states=False,
                capacity=100
            ),
            optimizer=dict(
                type='adam',
                learning_rate=1e-2
            ),
            baseline_mode='states',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                num_steps=5
            )
        )
        self.base_test_pass(
            name='states-baseline',
            environment=environment,
            network=network,
            **config
        )

    def test_network_baseline(self):
        environment = MinimalTest(specification={'int': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        config = dict(
            update_mode=dict(
                unit='episodes',
                batch_size=4,
                frequency=4
            ),
            memory=dict(
                type='latest',
                include_next_states=False,
                capacity=100
            ),
            optimizer=dict(
                type='adam',
                learning_rate=1e-2
            ),
            baseline_mode='network',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                num_steps=5
            )
        )
        self.base_test_pass(
            name='network-baseline',
            environment=environment,
            network=network,
            **config
        )

    def test_baseline_no_optimizer(self):
        environment = MinimalTest(specification={'int': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        config = dict(
            update_mode=dict(
                unit='episodes',
                batch_size=4,
                frequency=4
            ),
            memory=dict(
                type='latest',
                include_next_states=False,
                capacity=100
            ),
            optimizer=dict(
                type='adam',
                learning_rate=1e-2
            ),
            baseline_mode='states',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            )
        )
        self.base_test_pass(
            name='baseline-no-optimizer',
            environment=environment,
            network=network,
            **config
        )

    def test_gae_baseline(self):
        environment = MinimalTest(specification={'int': ()})
        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        config = dict(
            update_mode=dict(
                unit='episodes',
                batch_size=4,
                frequency=4
            ),
            memory=dict(
                type='latest',
                include_next_states=False,
                capacity=100
            ),
            optimizer=dict(
                type='adam',
                learning_rate=1e-2
            ),
            baseline_mode='states',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                num_steps=5
            ),
            gae_lambda=0.95
        )
        self.base_test_pass(
            name='gae-baseline',
            environment=environment,
            network=network,
            **config
        )

    def test_multi_baseline(self):

        class CustomNetwork(LayerBasedNetwork):

            def __init__(self, scope='layerbased-network', summary_labels=()):
                super(CustomNetwork, self).__init__(scope=scope, summary_labels=summary_labels)

                self.layer_bool1 = Dense(size=16, scope='state-bool1')
                self.add_layer(layer=self.layer_bool1)
                self.layer_bool2 = Dense(size=16, scope='state-bool2')
                self.add_layer(layer=self.layer_bool2)

                self.layer_int1 = Dense(size=16, scope='state-int1')
                self.add_layer(layer=self.layer_int1)
                self.layer_int2 = Dense(size=16, scope='state-int2')
                self.add_layer(layer=self.layer_int2)

                self.layer_float1 = Dense(size=16, scope='state-float1')
                self.add_layer(layer=self.layer_float1)
                self.layer_float2 = Dense(size=16, scope='state-float2')
                self.add_layer(layer=self.layer_float2)

                self.layer_bounded1 = Dense(size=16, scope='state-bounded1')
                self.add_layer(layer=self.layer_bounded1)
                self.layer_bounded2 = Dense(size=16, scope='state-bounded2')
                self.add_layer(layer=self.layer_bounded2)

            def tf_apply(self, x, internals, update, return_internals=False):
                x0 = self.layer_bool2.apply(x=self.layer_bool1.apply(x=x['bool'], update=update), update=update)
                x1 = self.layer_int2.apply(x=self.layer_int1.apply(x=x['int'], update=update), update=update)
                x2 = self.layer_float2.apply(x=self.layer_float1.apply(x=x['float'], update=update), update=update)
                x3 = self.layer_bounded2.apply(x=self.layer_bounded1.apply(x=x['bounded'], update=update), update=update)
                x = x0 * x1 * x2 * x3
                return (x, dict()) if return_internals else x

        environment = MinimalTest(
            specification={'bool': (), 'int': (2,), 'float': (1, 1), 'bounded': (1,)}
        )
        config = dict(
            update_mode=dict(
                unit='episodes',
                batch_size=4,
                frequency=4
            ),
            memory=dict(
                type='latest',
                include_next_states=False,
                capacity=100
            ),
            optimizer=dict(
                type='adam',
                learning_rate=1e-2
            ),
            baseline_mode='states',
            baseline=dict(
                type='aggregated',
                baselines={
                    'bool': dict(
                        type='mlp',
                        sizes=[32, 32]
                    ),
                    'int': dict(
                        type='mlp',
                        sizes=[32, 32]
                    ),
                    'float': dict(
                        type='mlp',
                        sizes=[32, 32]
                    ),
                    'bounded': dict(
                        type='mlp',
                        sizes=[32, 32]
                    )
                }
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                num_steps=5
            )
        )

        self.base_test_pass(
            name='multi-baseline',
            environment=environment,
            network=CustomNetwork,
            **config
        )
