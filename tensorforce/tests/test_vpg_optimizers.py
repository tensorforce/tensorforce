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


class TestVPGOptimizers(UnittestBase, unittest.TestCase):

    agent = VPGAgent

    def test_adam(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(
            name='adam', environment=environment, network=network, config=config
        )

    def test_clipped_step(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                type='clipped_step',
                optimizer=dict(type='adam', learning_rate=1e-3),
                clipping_value=1e-2
            )
        )

        self.unittest(
            name='clipped-step', environment=environment, network=network, config=config
        )

    def test_evolutionary(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(type='evolutionary', learning_rate=1e-3)
        )

        self.unittest(
            name='evolutionary', environment=environment, network=network, config=config
        )

    def test_kfac(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(type='kfac', learning_rate=1e-3)
        )

        self.unittest(
            name='kfac', environment=environment, network=network, config=config
        )

    def test_multi_step(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                type='multi_step',
                optimizer=dict(type='adam', learning_rate=1e-3)
            )
        )

        self.unittest(
            name='multi-step', environment=environment, network=network, config=config
        )

    def test_natural_gradient(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(type='natural_gradient', learning_rate=1e-3)
        )

        self.unittest(
            name='natural-gradient', environment=environment, network=network, config=config
        )

    def test_optimized_step(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                type='optimized_step',
                optimizer=dict(type='adam', learning_rate=1e-3)
            )
        )

        self.unittest(
            name='optimized-step', environment=environment, network=network, config=config
        )

    def test_subsampling_step(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            optimizer=dict(
                type='subsampling_step',
                optimizer=dict(type='adam', learning_rate=1e-3),
                fraction=0.5
            )
        )

        self.unittest(
            name='subsampling-step', environment=environment, network=network, config=config
        )
