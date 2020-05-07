# Copyright 2020 Tensorforce Team. All Rights Reserved.
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


class TestAgents(UnittestBase, unittest.TestCase):

    agent = dict()

    def test_a2c(self):
        self.start_tests(name='A2C')
        self.unittest(
            agent='a2c', batch_size=4, network=dict(type='auto', size=8, depth=1, rnn=2),
            critic_network=dict(type='auto', size=7, depth=1, rnn=1)
        )

    def test_ac(self):
        self.start_tests(name='AC')
        self.unittest(
            agent='ac', batch_size=4, network=dict(type='auto', size=8, depth=1, rnn=2),
            critic_network=dict(type='auto', size=7, depth=1, rnn=1)
        )

    def test_constant(self):
        self.start_tests(name='Constant')
        self.unittest(agent='constant')

    def test_dpg(self):
        self.start_tests(name='DPG')
        self.unittest(
            actions=dict(type='float', shape=(), min_value=-1.0, max_value=1.0),
            agent='dpg', memory=100, batch_size=4,
            network=dict(type='auto', size=8, depth=1, rnn=2),
            # TODO: baseline horizon has to be equal to policy horizon
            critic_network=dict(type='auto', size=7, depth=1, rnn=2)
        )

    def test_double_dqn(self):
        self.start_tests(name='DoubleDQN')
        # TODO: RNN is not supported
        self.unittest(
            actions=dict(type='int', shape=(2,), num_values=4),
            agent='double_dqn', memory=100, batch_size=4
        )

    def test_dqn(self):
        self.start_tests(name='DQN')
        # TODO: RNN is not supported
        self.unittest(
            actions=dict(type='int', shape=(2,), num_values=4),
            agent='dqn', memory=100, batch_size=4
        )

    def test_dueling_dqn(self):
        self.start_tests(name='DuelingDQN')
        # TODO: RNN is not supported
        self.unittest(
            actions=dict(type='int', shape=(2,), num_values=4),
            agent='dueling_dqn', memory=100, batch_size=4
        )

    def test_ppo(self):
        self.start_tests(name='PPO')
        self.unittest(agent='ppo', batch_size=2, network=dict(type='auto', size=8, depth=1, rnn=2))

    def test_random(self):
        self.start_tests(name='Random')
        self.unittest(agent='random')

    def test_tensorforce(self):
        self.start_tests(name='Tensorforce')
        # Explicit
        self.unittest(
            dict(type='float', shape=(1,)), dict(type='float', shape=()),
            agent='tensorforce', policy=dict(network=dict(type='auto', size=8, depth=1, rnn=2)),
            update=4, objective='policy_gradient', reward_estimation=dict(horizon=3)
        )
        # Implicit
        self.unittest(
            policy=dict(network=dict(type='auto', size=8, depth=1, rnn=2)), update=4,
            objective='policy_gradient', reward_estimation=dict(horizon=3)
        )

    def test_trpo(self):
        self.start_tests(name='TRPO')
        self.unittest(agent='trpo', batch_size=2, network=dict(type='auto', size=8, depth=1, rnn=2))

    def test_vpg(self):
        self.start_tests(name='VPG')
        self.unittest(agent='vpg', batch_size=2, network=dict(type='auto', size=8, depth=1, rnn=2))
