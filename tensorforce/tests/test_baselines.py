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

        config = dict(
            baseline_mode='states',
            baseline=dict(type='network', network='auto'),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(name='baseline-states', states=states, actions=actions, **config)

    def test_baseline_multistates(self):
        states = dict(
            bool_state=dict(type='bool', shape=(1,)),
            int_state=dict(type='int', shape=(2,), num_values=4),
            float_state=dict(type='float', shape=(1, 1, 2)),
            bounded_state=dict(type='float', shape=(), min_value=-0.5, max_value=0.5)
        )

        actions = dict(type='int', shape=(), num_values=3)

        config = dict(
            baseline_mode='states',
            baseline=dict(type='network', network='auto'),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(name='network-baseline', states=states, actions=actions, **config)

    def test_baseline_network(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        config = dict(
            baseline_mode='network',
            baseline=dict(type='network', network='auto'),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3)
        )

        self.unittest(name='baseline-network', states=states, actions=actions, **config)

    def test_baseline_no_optimizer(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        config = dict(
            baseline_mode='states',
            baseline=dict(type='network', network='auto')
        )

        self.unittest(name='baseline-no-optimizer', states=states, actions=actions, **config)

    def test_baseline_gae(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        config = dict(
            baseline_mode='states',
            baseline=dict(type='network', network='auto'),
            baseline_optimizer=dict(type='adam', learning_rate=1e-3),
            gae_lambda=0.95
        )

        self.unittest(name='baseline-gae', states=states, actions=actions, **config)
