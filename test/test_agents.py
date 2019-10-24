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

from test.unittest_base import UnittestBase
from tensorforce.agents import Agent
from tensorforce.environments import Environment


class TestAgents(UnittestBase, unittest.TestCase):

    agent = dict()
    require_observe = True

    def test_ac(self):
        self.start_tests(name='AC')
        self.unittest(agent='ac', batch_size=2)

    def test_a2c(self):
        self.start_tests(name='A2C')
        self.unittest(agent='a2c', batch_size=2)

    def test_dpg(self):
        self.start_tests(name='DPG')
        self.unittest(
            actions=dict(type='float', shape=()), exclude_bool_action=True,
            exclude_int_action=True, agent='dpg',
            network=dict(type='auto', size=8, internal_rnn=False), batch_size=4,
            critic_network=dict(type='auto', size=8, internal_rnn=False)
            # TODO: shouldn't be necessary!
        )

    def test_dqn(self):
        self.start_tests(name='DQN')
        self.unittest(agent='dqn', batch_size=4)

    def test_dueling_dqn(self):
        self.start_tests(name='DuelingDQN')
        self.unittest(agent='dueling_dqn', batch_size=4)

    def test_ppo(self):
        self.start_tests(name='PPO')
        self.unittest(
            agent='ppo', network=dict(type='auto', size=8, internal_rnn=False), batch_size=2
            # TODO: shouldn't be necessary!  # TODO: shouldn't be necessary!
        )

    def test_trpo(self):
        self.start_tests(name='TRPO')
        self.unittest(
            agent='trpo', network=dict(type='auto', size=8, internal_rnn=False), batch_size=2
            # TODO: shouldn't be necessary!  # TODO: shouldn't be necessary!
        )

    def test_vpg(self):
        self.start_tests(name='VPG')
        self.unittest(agent='vpg', batch_size=2)
