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


class TestVPGMemories(UnittestBase, unittest.TestCase):

    agent = VPGAgent

    def test_latest_timesteps(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='timesteps', batch_size=2),
            memory=dict(type='latest', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='latest-timesteps', environment=environment, network=network, config=config
        )

    def test_latest_episodes(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='episodes', batch_size=2),
            memory=dict(type='latest', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='latest-episodes', environment=environment, network=network, config=config
        )

    def test_latest_sequences(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='sequences', batch_size=2),
            memory=dict(type='latest', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='latest-sequences', environment=environment, network=network, config=config
        )

    def test_replay_timesteps(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='timesteps', batch_size=2),
            memory=dict(type='replay', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='replay-timesteps', environment=environment, network=network, config=config
        )

    def test_replay_episodes(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='episodes', batch_size=2),
            memory=dict(type='replay', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='replay-episodes', environment=environment, network=network, config=config
        )

    def test_replay_sequences(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='sequences', batch_size=2),
            memory=dict( type='replay', include_next_states=False, capacity=100)
        )

        self.unittest(
            name='replay-sequences', environment=environment, network=network, config=config
        )

    def test_prioritized_replay_timesteps(self):
        environment = UnittestEnvironment(
            states=dict(type='float', shape=(1,)), actions=dict(type='float', shape=())
        )

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        config = dict(
            update_mode=dict(unit='timesteps', batch_size=2),
            memory=dict(
                type='prioritized_replay', include_next_states=False, capacity=100, buffer_size=100
            )
        )

        self.unittest(
            name='prioritized-replay-timesteps', environment=environment, network=network, config=config
        )
