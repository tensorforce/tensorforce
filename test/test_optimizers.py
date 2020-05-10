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

    def test_update_modifier_wrapper(self):
        self.start_tests(name='update-modifier-wrapper')

        self.unittest(optimizer=dict(
            optimizer='adam', learning_rate=1e-3, multi_step=5, subsampling_fraction=0.5,
            clipping_threshold=1e-2, optimizing_iterations=3
        ))

    def test_natural_gradient(self):
        self.start_tests(name='natural-gradient')

        self.unittest(optimizer=dict(type='natural_gradient', learning_rate=1e-3))

    def test_plus(self):
        self.start_tests(name='plus')

        optimizer = dict(
            type='plus', optimizer1=dict(type='adam', learning_rate=1e-3),
            optimizer2=dict(type='adagrad', learning_rate=1e-3)
        )
        self.unittest(optimizer=optimizer)

    def test_synchronization(self):
        self.start_tests(name='synchronization')

        self.unittest(
            policy=dict(network=dict(type='auto', size=8, depth=1, rnn=2)),
            optimizer='synchronization',
            baseline_policy=dict(network=dict(type='auto', size=8, depth=1, rnn=1)),
            baseline_optimizer='adam', baseline_objective='policy_gradient'
        )

    def test_tf_optimizer(self):
        self.start_tests(name='tf-optimizer')

        self.unittest(optimizer=dict(type='adam', learning_rate=1e-3))

        try:
            import tensorflow_addons as tfa

            self.unittest(optimizer=dict(
                type='radam', learning_rate=1e-3, decoupled_weight_decay=0.01, lookahead=True,
                moving_average=True
            ))

        except ModuleNotFoundError:
            pass
