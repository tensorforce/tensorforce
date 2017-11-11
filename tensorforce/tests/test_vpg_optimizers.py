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
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.tests.base_test import BaseTest


class TestVPGOptimizers(BaseTest, unittest.TestCase):

    agent = VPGAgent
    deterministic = False

    # TODO: Tests for other TensorFlow optimizers, necessary?

    def test_adam(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
            batch_size=8,
            optimizer=dict(
                type='adam',
                learning_rate=1e-3
            )
        )
        self.base_test(name='adam', environment=environment, network_spec=network_spec, config=config)

    def test_evolutionary(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
            batch_size=8,
            optimizer=dict(
                type='evolutionary',
                learning_rate=1e-2
            )
        )
        self.base_test(name='evolutionary', environment=environment, network_spec=network_spec, config=config)

    def test_natural_gradient(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
            batch_size=8,
            optimizer=dict(
                type='natural_gradient',
                learning_rate=1e-3
            )
        )
        self.base_test(name='natural-gradient', environment=environment, network_spec=network_spec, config=config)

    def test_multi_step(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
            batch_size=8,
            optimizer=dict(
                type='multi_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-4
                )
            )
        )
        self.base_test(name='multi-step', environment=environment, network_spec=network_spec, config=config)

    def test_optimized_step(self):
        environment = MinimalTest(specification=[('int', ())])
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
            batch_size=8,
            optimizer=dict(
                type='optimized_step',
                optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                )
            )
        )
        self.base_test(name='optimized-step', environment=environment, network_spec=network_spec, config=config)
