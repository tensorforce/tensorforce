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


class TestOptimizers(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))

    def test_adam(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='adam', states=states, actions=actions, network=network, **config
        )

    def test_clipped_step(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                type='clipped_step', optimizer=dict(type='adam', learning_rate=1e-3),
                clipping_value=1e-2
            )
        )

        self.unittest(
            name='clipped-step', states=states, actions=actions, network=network, **config
        )

    def test_evolutionary(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(optimizer=dict(type='evolutionary', learning_rate=1e-3))

        self.unittest(
            name='evolutionary', states=states, actions=actions, network=network, **config
        )

    def test_meta_optimizer_wrapper(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                optimizer='adam', learning_rate=1e-3, multi_step=5, subsampling_fraction=0.5,
                clipping_value=1e-2, optimized_iterations=3
            )
        )

        self.unittest(
            name='meta-optimizer-wrapper', states=states, actions=actions, network=network,
            **config
        )

    def test_multi_step(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                type='multi_step', optimizer=dict(type='adam', learning_rate=1e-3), num_steps=10
            )
        )

        self.unittest(
            name='multi-step', states=states, actions=actions, network=network, **config
        )

    def test_natural_gradient(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(type='natural_gradient', learning_rate=1e-3)
        )

        self.unittest(
            name='natural-gradient', states=states, actions=actions, network=network, **config
        )

    def test_optimized_step(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(type='optimized_step', optimizer=dict(type='adam', learning_rate=1e-3))
        )

        self.unittest(
            name='optimized-step', states=states, actions=actions, network=network, **config
        )

    def test_subsampling_step(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                type='subsampling_step', optimizer=dict(type='adam', learning_rate=1e-3),
                fraction=0.5
            )
        )

        self.unittest(
            name='subsampling-step', states=states, actions=actions, network=network, **config
        )
