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


class TestRewardEstimation(UnittestBase, unittest.TestCase):

    exclude_bounded_action = True  # TODO: shouldn't be necessary!
    require_observe = True

    def test_reward_estimation(self):
        self.start_tests()

        # zero horizon
        reward_estimation = dict(horizon=0)
        self.unittest(reward_estimation=reward_estimation)

        # horizon longer than episode
        reward_estimation = dict(horizon=10)
        self.unittest(timestep_range=(1, 5), reward_estimation=reward_estimation)

        # discount
        reward_estimation = dict(horizon=2, discount=0.99)
        self.unittest(reward_estimation=reward_estimation)

        # estimate horizon early
        reward_estimation = dict(horizon=2, estimate_horizon='early')
        baseline_policy = 'equal'
        self.unittest(reward_estimation=reward_estimation, baseline_policy=baseline_policy)

        reward_estimation = dict(horizon=2, estimate_horizon='early', estimate_actions=True)
        baseline_policy = 'equal'
        self.unittest(reward_estimation=reward_estimation, baseline_policy=baseline_policy)

        reward_estimation = dict(horizon=2, estimate_horizon='early', estimate_terminal=True)
        baseline_policy = 'equal'
        self.unittest(reward_estimation=reward_estimation, baseline_policy=baseline_policy)

        # estimate horizon late
        reward_estimation = dict(horizon=2, estimate_horizon='late')
        baseline_network = dict(type='auto', size=8, internal_rnn=2)
        baseline_objective = 'equal'
        baseline_optimizer = 'equal'
        self.unittest(
            reward_estimation=reward_estimation, baseline_network=baseline_network,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='late', estimate_actions=True)
        baseline_network = dict(type='auto', size=8, internal_rnn=2)
        baseline_objective = 'equal'
        baseline_optimizer = 'equal'
        self.unittest(
            reward_estimation=reward_estimation, baseline_network=baseline_network,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_horizon='late', estimate_terminal=True)
        baseline_network = dict(type='auto', size=8, internal_rnn=2)
        baseline_objective = 'equal'
        baseline_optimizer = 'equal'
        self.unittest(
            reward_estimation=reward_estimation, baseline_network=baseline_network,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        # estimate advantage
        reward_estimation = dict(horizon=2, estimate_advantage=True)
        baseline_network = dict(type='auto', size=8, internal_rnn=2)
        baseline_objective = 'equal'
        baseline_optimizer = 'equal'
        self.unittest(
            reward_estimation=reward_estimation, baseline_network=baseline_network,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        reward_estimation = dict(horizon=2, estimate_actions=True, estimate_advantage=True)
        baseline_network = dict(type='auto', size=8, internal_rnn=2)
        baseline_objective = 'equal'
        baseline_optimizer = 'equal'
        self.unittest(
            reward_estimation=reward_estimation, baseline_network=baseline_network,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )
