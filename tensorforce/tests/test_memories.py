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


class TestMemories(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))

    def test_latest_timesteps(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='timesteps', batch_size=2),
            memory=dict(type='latest', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='latest-timesteps', states=states, actions=actions, network=network, **config
        )

    def test_latest_episodes(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='episodes', batch_size=2),
            memory=dict(type='latest', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='latest-episodes', states=states, actions=actions, network=network, **config
        )

    def test_latest_sequences(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='sequences', sequence_length=4, batch_size=2),
            memory=dict(type='latest', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='latest-sequences', states=states, actions=actions, network=network, **config
        )

    def test_replay_timesteps(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='timesteps', batch_size=2),
            memory=dict(type='replay', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='replay-timesteps', states=states, actions=actions, network=network, **config
        )

    def test_replay_episodes(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='episodes', batch_size=2),
            memory=dict(type='replay', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='replay-episodes', states=states, actions=actions, network=network, **config
        )

    def test_replay_sequences(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='sequences', sequence_length=4, batch_size=2),
            memory=dict(type='replay', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='replay-sequences', states=states, actions=actions, network=network, **config
        )

    def broken_test_prioritized_replay_timesteps(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='timesteps', batch_size=2),
            memory=dict(
                type='prioritized_replay', include_next_states=False, capacity=100, buffer_size=100
            )
        )

        self.unittest(
            name='prioritized-replay-timesteps', states=states, actions=actions, network=network, **config
        )
