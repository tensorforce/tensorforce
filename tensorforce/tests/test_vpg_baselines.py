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
from tensorforce.agents import VPGAgent
from tensorforce.core.networks import Dense, LayerBasedNetwork
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.tests.base_test import BaseTest


class TestVPGBaselines(BaseTest, unittest.TestCase):

    agent = VPGAgent
    deterministic = False

    def test_states_baseline(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
            batch_size=8,
            baseline_mode='states',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=0.001
                ),
                num_steps=5
            )
        )
        self.base_test(
            name='states-baseline',
            environment=environment,
            network_spec=network_spec,
            **kwargs
        )

    def test_network_baseline(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        kwargs = dict(
            batch_size=8,
            baseline_mode='network',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=0.001
                ),
                num_steps=5
            )
        )
        self.base_test(
            name='network-baseline',
            environment=environment,
            network_spec=network_spec,
            **kwargs
        )

    def test_baseline_no_optimizer(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        kwargs = dict(
            batch_size=8,
            baseline_mode='states',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            )
        )
        self.base_test(
            name='baseline-no-optimizer',
            environment=environment,
            network_spec=network_spec,
            **kwargs
        )

    def test_gae_baseline(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
            batch_size=8,
            baseline_mode='states',
            baseline=dict(
                type='mlp',
                sizes=[32, 32]
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=0.001
                ),
                num_steps=5
            ),
            gae_lambda=0.95,
            normalize_rewards=True
        )
        self.base_test(
            name='gae-baseline',
            environment=environment,
            network_spec=network_spec,
            **kwargs
        )

    def test_multi_baseline(self):

        class CustomNetwork(LayerBasedNetwork):

            def __init__(self, scope='layerbased-network', summary_labels=()):
                super(CustomNetwork, self).__init__(scope=scope, summary_labels=summary_labels)

                self.layer01 = Dense(size=32, scope='state0-1')
                self.add_layer(layer=self.layer01)
                self.layer02 = Dense(size=32, scope='state0-2')
                self.add_layer(layer=self.layer02)

                self.layer11 = Dense(size=32, scope='state1-1')
                self.add_layer(layer=self.layer11)
                self.layer12 = Dense(size=32, scope='state1-2')
                self.add_layer(layer=self.layer12)

                self.layer21 = Dense(size=32, scope='state2-1')
                self.add_layer(layer=self.layer21)
                self.layer22 = Dense(size=32, scope='state2-2')
                self.add_layer(layer=self.layer22)

                self.layer31 = Dense(size=32, scope='state3-1')
                self.add_layer(layer=self.layer31)
                self.layer32 = Dense(size=32, scope='state3-2')
                self.add_layer(layer=self.layer32)

            def tf_apply(self, x, internals, update, return_internals=False):
                x0 = self.layer02.apply(x=self.layer01.apply(x=x['state0'], update=update), update=update)
                x1 = self.layer12.apply(x=self.layer11.apply(x=x['state1'], update=update), update=update)
                x2 = self.layer22.apply(x=self.layer21.apply(x=x['state2'], update=update), update=update)
                x3 = self.layer32.apply(x=self.layer31.apply(x=x['state3'], update=update), update=update)
                x = x0 * x1 * x2 * x3
                return (x, list()) if return_internals else x

        environment = MinimalTest(
            specification=[('bool', ()), ('int', (2,)), ('float', (1, 1)), ('bounded-float', (1,))]
        )
        kwargs = dict(
            batch_size=8,
            baseline_mode='states',
            baseline=dict(
                type='aggregated',
                baselines=dict(
                    state0=dict(
                        type='mlp',
                        sizes=[32, 32]
                    ),
                    state1=dict(
                        type='mlp',
                        sizes=[32, 32]
                    ),
                    state2=dict(
                        type='mlp',
                        sizes=[32, 32]
                    ),
                    state3=dict(
                        type='mlp',
                        sizes=[32, 32]
                    )
                )
            ),
            baseline_optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=0.001
                ),
                num_steps=5
            )
        )

        self.base_test(
            name='multi-baseline',
            environment=environment,
            network_spec=CustomNetwork,
            **kwargs
        )
