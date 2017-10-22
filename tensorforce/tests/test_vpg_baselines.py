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

from tensorforce import Configuration
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
        config = Configuration(
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
        self.base_test(name='states-baseline', environment=environment, network_spec=network_spec, config=config)

    def test_network_baseline(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        config = Configuration(
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
        self.base_test(name='network-baseline', environment=environment, network_spec=network_spec, config=config)

    def test_gae_baseline(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        config = Configuration(
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
        self.base_test(name='gae-baseline', environment=environment, network_spec=network_spec, config=config)

    def test_multi_baseline(self):

        class CustomNetwork(LayerBasedNetwork):
            def tf_apply(self, x, internals, return_internals=False):
                layer01 = Dense(size=32, scope='state0-1')
                self.add_layer(layer=layer01)
                layer02 = Dense(size=32, scope='state0-2')
                self.add_layer(layer=layer02)
                x0 = layer02.apply(x=layer01.apply(x=x['state0']))
                layer11 = Dense(size=32, scope='state1-1')
                self.add_layer(layer=layer11)
                layer12 = Dense(size=32, scope='state1-2')
                self.add_layer(layer=layer12)
                x1 = layer12.apply(x=layer11.apply(x=x['state1']))
                layer21 = Dense(size=32, scope='state2-1')
                self.add_layer(layer=layer21)
                layer22 = Dense(size=32, scope='state2-2')
                self.add_layer(layer=layer22)
                x2 = layer22.apply(x=layer21.apply(x=x['state2']))
                layer31 = Dense(size=32, scope='state3-1')
                self.add_layer(layer=layer31)
                layer32 = Dense(size=32, scope='state3-2')
                self.add_layer(layer=layer32)
                x3 = layer32.apply(x=layer31.apply(x=x['state3']))
                x = x0 * x1 * x2 * x3
                return (x, list()) if return_internals else x

        environment = MinimalTest(
            specification=[('bool', ()), ('int', (2,)), ('float', (1,)), ('bounded-float', (1, 1))]
        )
        config = Configuration(
            batch_size=8,
            baseline_mode='states',
            # baseline=dict(
            #     type='mlp',
            #     sizes=[32, 32]
            # ),
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
        self.base_test(name='multi-baseline', environment=environment, network_spec=CustomNetwork, config=config)
