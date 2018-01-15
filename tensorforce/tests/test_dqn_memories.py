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
from tensorforce.agents import DQNAgent
from tensorforce.environments.minimal_test import MinimalTest


class TestDQNMemories(BaseTest, unittest.TestCase):

    agent = DQNAgent
    deterministic = True

    def test_replay(self):
        environment = MinimalTest(specification={'int': ()})
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        config = dict(
            memory=dict(
                type='replay',
                capacity=1000
            ),
            repeat_update=4,
            batch_size=32,
            first_update=64,
            target_sync_frequency=10
        )

        self.base_test_pass(
            name='replay',
            environment=environment,
            network_spec=network_spec,
            **config
        )

    def test_prioritized_replay(self):
        environment = MinimalTest(specification={'int': ()})
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]

        config = dict(
            memory=dict(
                type='prioritized_replay',
                capacity=1000
            ),
            repeat_update=4,
            batch_size=32,
            first_update=64,
            target_sync_frequency=10
        )

        self.base_test_pass(
            name='prioritized-replay',
            environment=environment,
            network_spec=network_spec,
            **config
        )

    def test_naive_prioritized_replay(self):
        environment = MinimalTest(specification={'int': ()})
        network_spec = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        config = dict(
            memory=dict(
                type='naive_prioritized_replay',
                capacity=1000
            ),
            repeat_update=4,
            batch_size=32,
            first_update=64,
            target_sync_frequency=10
        )

        self.base_test_pass(
            name='naive-prioritized-replay',
            environment=environment,
            network_spec=network_spec,
            **config
        )
