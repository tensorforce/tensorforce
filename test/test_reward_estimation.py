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


class TestRewardEstimation(UnittestBase, unittest.TestCase):

    agent = dict(
        policy=dict(network=dict(type='auto', size=8, depth=1, rnn=2)),
        update=dict(unit='episodes', batch_size=1),
        objective='policy_gradient'
    )

    def test_no_horizon_estimate(self):
        self.start_tests(name='no horizon estimate')

        # zero horizon
        reward_estimation = dict(horizon=0, discount=0.99, estimate_horizon=False)
        self.unittest(reward_estimation=reward_estimation)

        # horizon longer than episode
        reward_estimation = dict(horizon=10, discount=0.99, estimate_horizon=False)
        self.unittest(reward_estimation=reward_estimation)

    def test_early_horizon_estimate(self):
        self.start_tests(name='early horizon estimate')

        # TODO: action value doesn't exist for Beta
        policy = dict(
            network=dict(type='auto', size=8, depth=1, rnn=2), use_beta_distribution=False
        )
        reward_estimation = dict(horizon=2, estimate_horizon='early', estimate_action_values=True)
        self.unittest(policy=policy, reward_estimation=reward_estimation)

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        # Implicit baseline_policy = policy
        baseline_optimizer = 'adam'
        baseline_objective = 'value'
        self.unittest(
            reward_estimation=reward_estimation, baseline_optimizer=baseline_optimizer,
            baseline_objective=baseline_objective
        )

        reward_estimation = dict(horizon=2, estimate_horizon='early', estimate_terminals=True)
        baseline_policy = dict(network=dict(type='auto', size=7, depth=1, rnn=1))
        # Implicit baseline_optimizer = 1.0
        baseline_objective = 'value'
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective
        )

        reward_estimation = dict(
            horizon=2, estimate_horizon='early', estimate_action_values=True,
            estimate_terminals=True
        )
        # TODO: action value doesn't exist for Beta
        baseline_policy = dict(
            network=dict(type='auto', size=7, depth=1, rnn=1), use_beta_distribution=False
        )
        baseline_optimizer = 'adam'
        baseline_objective = dict(type='value', value='action')
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )

    def test_late_horizon_estimate(self):
        self.start_tests(name='late horizon estimate')

        reward_estimation = dict(horizon=2, estimate_horizon='late')
        # Implicit baseline_policy = policy
        # Implicit baseline_optimizer = 1.0
        baseline_objective = 'value'
        self.unittest(reward_estimation=reward_estimation, baseline_objective=baseline_objective)

        reward_estimation = dict(horizon=2, estimate_horizon='late', estimate_action_values=True)
        # TODO: action value doesn't exist for Beta
        # TODO: baseline horizon has to be equal to policy horizon
        baseline_policy = dict(
            network=dict(type='auto', size=7, depth=1, rnn=2), use_beta_distribution=False
        )
        baseline_optimizer = 2.0
        baseline_objective = dict(type='value', value='action')
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )

        reward_estimation = dict(horizon=2, estimate_horizon='late', estimate_terminals=True)
        # Implicit baseline_policy = policy
        baseline_optimizer = 'adam'
        baseline_objective = 'value'
        self.unittest(
            reward_estimation=reward_estimation, baseline_optimizer=baseline_optimizer,
            baseline_objective=baseline_objective
        )

        reward_estimation = dict(
            horizon=2, estimate_horizon='late', estimate_action_values=True, estimate_terminals=True
        )
        # TODO: action value doesn't exist for Beta
        # TODO: baseline horizon has to be equal to policy horizon
        baseline_policy = dict(
            network=dict(type='auto', size=7, depth=1, rnn=2), use_beta_distribution=False
        )
        baseline_optimizer = 'adam'
        baseline_objective = dict(type='value', value='action')
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )

    def test_advantage_estimate(self):
        self.start_tests(name='advantage estimate')

        reward_estimation = dict(horizon=2, estimate_horizon=False, estimate_advantage=True)
        baseline_policy = dict(
            network=dict(type='auto', size=7, depth=1, rnn=1), use_beta_distribution=False
        )
        # Implicit advantage computation as part of loss
        self.unittest(reward_estimation=reward_estimation, baseline_policy=baseline_policy)

        # TODO: action value doesn't exist for Beta
        policy = dict(
            network=dict(type='auto', size=8, depth=1, rnn=2), use_beta_distribution=False
        )
        reward_estimation = dict(
            horizon=2, estimate_horizon='early', estimate_action_values=True,
            estimate_advantage=True
        )
        # Implicit baseline_policy = policy
        baseline_objective = 'value'
        self.unittest(
            policy=policy, reward_estimation=reward_estimation,
            baseline_objective=baseline_objective
        )

        reward_estimation = dict(
            horizon=2, estimate_horizon='late', estimate_terminals=True, estimate_advantage=True
        )
        baseline_policy = dict(network=dict(type='auto', size=7, depth=1, rnn=1))
        baseline_optimizer = 'adam'
        baseline_objective = 'value'
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )
