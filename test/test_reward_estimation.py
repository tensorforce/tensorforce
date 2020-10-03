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
        policy=dict(network=dict(type='auto', size=8, depth=1, rnn=2), distributions=dict(
            gaussian_action2=dict(type='gaussian', global_stddev=True), beta_action='beta'
        )), update=4, optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(
            horizon=3, estimate_advantage=True, predict_horizon_values='late',
            return_processing=dict(type='clipping', lower=-1.0, upper=1.0),
            advantage_processing='batch_normalization'
        ), l2_regularization=0.01, entropy_regularization=0.01,
        state_preprocessing='linear_normalization',
        reward_preprocessing=dict(type='clipping', lower=-1.0, upper=1.0),
        exploration=0.01, variable_noise=0.01,
        config=dict(eager_mode=True, create_debug_assertions=True, tf_log_level=20)
    )

    def test_no_horizon_estimate(self):
        self.start_tests(name='no horizon estimate')

        # shortest horizon
        reward_estimation = dict(
            horizon=1, discount=0.99, predict_horizon_values=False,
            return_processing='batch_normalization'
        )
        self.unittest(reward_estimation=reward_estimation)

        # horizon as long as episode
        reward_estimation = dict(
            horizon=10, discount=0.99, predict_horizon_values=False,
            return_processing='batch_normalization'
        )
        self.unittest(reward_estimation=reward_estimation)

        # episode horizon
        reward_estimation = dict(
            horizon='episode', discount=0.99, predict_horizon_values=False,
            return_processing='batch_normalization'
        )
        self.unittest(reward_estimation=reward_estimation)

    def test_early_horizon_estimate(self):
        self.start_tests(name='early horizon estimate')

        # TODO: action value doesn't exist for Beta
        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4),
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0)
        )
        reward_estimation = dict(
            horizon='episode', predict_horizon_values='early', predict_action_values=True,
            return_processing='batch_normalization'
        )
        # Implicit baseline = policy
        self.unittest(actions=actions, reward_estimation=reward_estimation, config=dict(
            buffer_observe=3, eager_mode=True, create_debug_assertions=True, tf_log_level=20
        ))

        # TODO: action value doesn't exist for Beta
        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4),
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0)
        )
        update = dict(unit='episodes', batch_size=1)
        reward_estimation = dict(
            horizon=3, predict_horizon_values='early', return_processing='batch_normalization'
        )
        # Implicit baseline = policy
        baseline_optimizer = dict(optimizer='adam', learning_rate=1e-3)
        baseline_objective = 'state_value'
        self.unittest(
            actions=actions, update=update, reward_estimation=reward_estimation,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective,
            config=dict(
                buffer_observe='episode', eager_mode=True, create_debug_assertions=True,
                tf_log_level=20
            )  # or 1?
        )

        reward_estimation = dict(
            horizon='episode', predict_horizon_values='early', predict_terminal_values=True,
            return_processing='batch_normalization'
        )
        baseline = dict(network=dict(type='auto', size=7, depth=1, rnn=1))
        # Implicit baseline_optimizer = 1.0
        baseline_objective = 'state_value'
        self.unittest(
            reward_estimation=reward_estimation, baseline=baseline,
            baseline_objective=baseline_objective
        )

        # Action-value baseline compatible with discrete actions
        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4)
        )
        reward_estimation = dict(
            horizon=3, predict_horizon_values='early', predict_action_values=True,
            predict_terminal_values=True, return_processing='batch_normalization'
        )
        baseline = dict(network=dict(type='auto', size=7, depth=1, rnn=1))
        baseline_optimizer = dict(optimizer='adam', learning_rate=1e-3)
        baseline_objective = 'action_value'
        self.unittest(
            actions=actions, reward_estimation=reward_estimation, baseline=baseline,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )

    def test_late_horizon_estimate(self):
        self.start_tests(name='late horizon estimate')

        # TODO: action value doesn't exist for Beta
        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4),
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0)
        )
        reward_estimation = dict(
            horizon=3, predict_horizon_values='late', return_processing='batch_normalization'
        )
        # Implicit baseline = policy
        # Implicit baseline_optimizer = 1.0
        baseline_objective = 'state_value'
        self.unittest(
            actions=actions, reward_estimation=reward_estimation,
            baseline_objective=baseline_objective
        )

        # Action-value baseline compatible with discrete actions
        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4)
        )
        reward_estimation = dict(
            horizon=3, predict_horizon_values='late', predict_action_values=True,
            return_processing='batch_normalization'
        )
        # TODO: baseline horizon has to be equal to policy horizon
        baseline = dict(network=dict(type='auto', size=7, depth=1, rnn=2))
        baseline_optimizer = 2.0
        baseline_objective = 'action_value'
        self.unittest(
            actions=actions, reward_estimation=reward_estimation, baseline=baseline,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )

        # TODO: state value doesn't exist for Beta
        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4),
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0)
        )
        reward_estimation = dict(
            horizon=3, predict_horizon_values='late', predict_terminal_values=True,
            return_processing='batch_normalization'
        )
        # Implicit baseline = policy
        baseline_optimizer = dict(optimizer='adam', learning_rate=1e-3)
        baseline_objective = 'state_value'
        self.unittest(
            actions=actions, reward_estimation=reward_estimation,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )

        reward_estimation = dict(
            horizon=3, predict_horizon_values='late', predict_action_values=True,
            predict_terminal_values=True, return_processing='batch_normalization'
        )
        # TODO: baseline horizon has to be equal to policy horizon
        # (Not specifying customized distributions since action value doesn't exist for Beta)
        baseline = dict(
            type='parametrized_distributions', network=dict(type='auto', size=7, depth=1, rnn=2)
        )
        baseline_optimizer = dict(optimizer='adam', learning_rate=1e-3)
        baseline_objective = 'action_value'
        self.unittest(
            reward_estimation=reward_estimation, baseline=baseline,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )

    def test_advantage_estimate(self):
        self.start_tests(name='advantage estimate')

        reward_estimation = dict(
            horizon=3, estimate_advantage=True, predict_horizon_values=False,
            return_processing=dict(type='clipping', lower=-1.0, upper=1.0),
            advantage_processing='batch_normalization'
        )
        baseline = dict(network=dict(type='auto', size=7, depth=1, rnn=1))
        # Implicit advantage computation as part of loss
        self.unittest(reward_estimation=reward_estimation, baseline=baseline)

        # TODO: action value doesn't exist for Beta
        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4),
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0)
        )
        reward_estimation = dict(
            horizon='episode', estimate_advantage=True, predict_horizon_values='early',
            predict_action_values=True,
            return_processing=dict(type='clipping', lower=-1.0, upper=1.0),
            advantage_processing='batch_normalization'
        )
        # Implicit baseline = policy
        # Implicit baseline_optimizer = 1.0
        baseline_objective = 'state_value'
        self.unittest(
            actions=actions, reward_estimation=reward_estimation,
            baseline_objective=baseline_objective
        )

        reward_estimation = dict(
            horizon=3, estimate_advantage=True, predict_horizon_values='late',
            predict_terminal_values=True,
            return_processing=dict(type='clipping', lower=-1.0, upper=1.0),
            advantage_processing='batch_normalization'
        )
        baseline = dict(network=dict(type='auto', size=7, depth=1, rnn=1))
        baseline_optimizer = dict(optimizer='adam', learning_rate=1e-3)
        baseline_objective = 'state_value'
        self.unittest(
            reward_estimation=reward_estimation, baseline=baseline,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective
        )
