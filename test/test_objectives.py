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

        actions = dict(
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0),
            beta_action=dict(type='float', shape=(), min_value=1.0, max_value=2.0)
        )
        # TODO: no-RNN restriction can be removed
        policy = dict(network=dict(type='auto', size=8, depth=1, rnn=False), distributions=dict(
            gaussian_action2=dict(type='gaussian', global_stddev=True), beta_action='beta'
        ))
        objective = 'deterministic_policy_gradient'
        reward_estimation = dict(
            horizon=3, estimate_advantage=True, predict_horizon_values='late',
            predict_action_values=True,
            return_processing=dict(type='clipping', lower=-1.0, upper=1.0),
            advantage_processing='batch_normalization'
        )
        baseline = dict(network=dict(type='auto', size=7, depth=1, rnn=False))
        baseline_objective = 'action_value'
        self.unittest(
            actions=actions, policy=policy, objective=objective,
            reward_estimation=reward_estimation, baseline=baseline,
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

        # State value does not affect advantage variables of main policy
        objective = 'state_value'
        self.unittest(actions=actions, baseline_objective=objective, entropy_regularization=0.0)

        policy = dict(network=dict(type='auto', size=8, depth=1, rnn=2))
        objective = dict(type='value', value='action')
        self.unittest(
            actions=actions, policy=policy, objective=objective, entropy_regularization=0.0
        )

        objective = dict(type='value', value='state', huber_loss=1.0)
        self.unittest(actions=actions, baseline_objective=objective, entropy_regularization=0.0)

        objective = dict(type='action_value', early_reduce=True)
        self.unittest(actions=actions, baseline_objective=objective, entropy_regularization=0.0)
