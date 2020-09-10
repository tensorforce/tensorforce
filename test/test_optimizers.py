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


class TestOptimizers(UnittestBase, unittest.TestCase):

    def test_evolutionary(self):
        self.start_tests(name='evolutionary')

        self.unittest(optimizer=dict(type='evolutionary', learning_rate=1e-3))

        self.unittest(optimizer=dict(type='evolutionary', learning_rate=1e-3, num_samples=5))

    def test_optimizer_wrapper(self):
        self.start_tests(name='optimizer-wrapper')

        self.unittest(optimizer=dict(
            optimizer='adam', learning_rate=1e-1, clipping_threshold=1e-2, multi_step=5,
            subsampling_fraction=0.5, linesearch_iterations=3, doublecheck_update=True
        ))

        self.unittest(optimizer=dict(optimizer='adam', subsampling_fraction=2))

    def test_natural_gradient(self):
        self.start_tests(name='natural-gradient')

        self.unittest(
            optimizer=dict(type='natural_gradient', learning_rate=1e-3, only_positive_updates=False)
        )

    def test_plus(self):
        self.start_tests(name='plus')

        optimizer = dict(
            type='plus', optimizer1=dict(type='adam', learning_rate=1e-3),
            optimizer2=dict(type='adagrad', learning_rate=1e-3)
        )
        self.unittest(optimizer=optimizer)

    def test_synchronization(self):
        self.start_tests(name='synchronization')

        actions = dict(
            bool_action=dict(type='bool', shape=(1,)),
            int_action=dict(type='int', shape=(2,), num_values=4),
            gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
            gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0)
        )
        # Requires same size, but can still vary RNN horizon
        baseline = dict(
            type='parametrized_distributions', network=dict(type='auto', size=8, depth=1, rnn=1),
            distributions=dict(gaussian_action2=dict(type='gaussian', global_stddev=True))
        )
        # Using policy_gradient here, since action_value is covered by DQN
        baseline_objective = 'policy_gradient'
        self.unittest(
            actions=actions, baseline=baseline,
            baseline_optimizer=dict(type='synchronization', update_weight=1.0),
            baseline_objective=baseline_objective
        )

        self.unittest(
            actions=actions, baseline=baseline,
            baseline_optimizer=dict(type='synchronization', update_weight=1.0, sync_frequency=2),
            baseline_objective=baseline_objective
        )

    def test_tf_optimizer(self):
        self.start_tests(name='tf-optimizer')

        self.unittest(optimizer=dict(type='adam', learning_rate=1e-3))

        self.unittest(optimizer=dict(type='adam', learning_rate=1e-3, gradient_norm_clipping=1.0))

        try:
            import tensorflow_addons as tfa

            self.unittest(optimizer=dict(
                type='tf_optimizer', optimizer='radam', learning_rate=1e-3,
                decoupled_weight_decay=0.01, lookahead=True, moving_average=True
            ))

        except ModuleNotFoundError:
            pass
        except TypeError:
            # TODO: temporary for version 0.11.1
            pass
