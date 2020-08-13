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


class TestObjectives(UnittestBase, unittest.TestCase):

    def test_deterministic_policy_gradient(self):
        self.start_tests(name='deterministic-policy-gradient')

        objective = 'deterministic_policy_gradient'
        baseline_objective = 'action_value'
        self.unittest(
            actions=dict(type='float', shape=(), min_value=1.0, max_value=2.0),
            policy=dict(network=dict(type='auto', size=8, depth=1, rnn=2)), objective=objective,
            baseline_objective=baseline_objective
        )

    def test_plus(self):
        self.start_tests(name='plus')

        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4),
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0)
        )
        objective = dict(type='plus', objective1='policy_gradient', objective2='action_value')
        self.unittest(actions=actions, objective=objective)

    def test_policy_gradient(self):
        self.start_tests(name='policy-gradient')

        objective = 'policy_gradient'
        self.unittest(objective=objective)

        objective = dict(type='policy_gradient', importance_sampling=True)
        self.unittest(objective=objective)

        objective = dict(type='policy_gradient', clipping_value=1.0)
        self.unittest(objective=objective)

        objective = dict(type='policy_gradient', importance_sampling=True, clipping_value=0.2)
        self.unittest(objective=objective)

        objective = dict(type='policy_gradient', early_reduce=True)
        self.unittest(objective=objective)

    def test_value(self):
        self.start_tests(name='value')

        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4)
        )
        policy = dict(network=dict(type='auto', size=8, depth=1, rnn=2))

        objective = 'state_value'
        self.unittest(
            actions=actions, policy=policy, objective=objective, entropy_regularization=0.0
        )

        objective = dict(type='value', value='action')
        self.unittest(
            actions=actions, policy=policy, objective=objective, entropy_regularization=0.0
        )

        objective = dict(type='value', value='state', huber_loss=1.0)
        self.unittest(
            actions=actions, policy=policy, objective=objective, entropy_regularization=0.0
        )

        objective = dict(type='action_value', early_reduce=True)
        self.unittest(
            actions=actions, policy=policy, objective=objective, entropy_regularization=0.0
        )
