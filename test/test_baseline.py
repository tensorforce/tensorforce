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


class TestBaseline(UnittestBase, unittest.TestCase):

    exclude_bounded_action = True  # TODO: shouldn't be necessary!
    require_observe = True

    def test_policy_as_baseline(self):
        self.start_tests(name='policy as baseline')

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = None
        baseline_objective = None
        baseline_optimizer = None
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = None
        baseline_objective = 'policy_gradient'
        baseline_optimizer = None
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = None
        baseline_objective = None
        baseline_optimizer = 'adam'
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = None
        baseline_objective = 'policy_gradient'
        baseline_optimizer = 'adam'
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

    def test_separate_baseline(self):
        self.start_tests(name='separate baseline')

        reward_estimation = dict(horizon=2, estimate_advantage=True)
        baseline_policy = dict(network=dict(type='auto', size=8, internal_rnn=2))
        baseline_objective = None
        baseline_optimizer = None
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = dict(network=dict(type='auto', size=8, internal_rnn=2))
        baseline_objective = 'policy_gradient'
        baseline_optimizer = None
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = dict(network=dict(type='auto', size=8, internal_rnn=1))
        baseline_objective = None
        baseline_optimizer = 'adam'
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = dict(network=dict(type='auto', size=8, internal_rnn=1))
        baseline_objective = 'policy_gradient'
        baseline_optimizer = 'adam'
        self.unittest(
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )
