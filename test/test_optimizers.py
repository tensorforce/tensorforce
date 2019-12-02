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


class TestOptimizers(UnittestBase, unittest.TestCase):

    require_observe = True
    num_updates = 10

    def test_evolutionary(self):
        self.start_tests(name='evolutionary')

        optimizer = dict(type='evolutionary', learning_rate=1e-3)
        self.unittest(optimizer=optimizer)

    def test_meta_optimizer_wrapper(self):
        self.start_tests(name='meta-optimizer-wrapper')

        optimizer = dict(
            type='meta_optimizer_wrapper', optimizer='adam', learning_rate=1e-3, multi_step=5,
            subsampling_fraction=0.5, clipping_threshold=1e-2, optimizing_iterations=3
        )
        self.unittest(
            optimizer=optimizer,
            policy=dict(network=dict(type='auto', size=8, internal_rnn=False))
            # TODO: shouldn't be necessary!
        )

        optimizer = dict(
            optimizer='adam', learning_rate=1e-3, multi_step=5, subsampling_fraction=0.5,
            clipping_threshold=1e-2, optimizing_iterations=3
        )
        self.unittest(
            optimizer=optimizer,
            policy=dict(network=dict(type='auto', size=8, internal_rnn=False))
            # TODO: shouldn't be necessary!
        )

    def test_natural_gradient(self):
        self.start_tests(name='natural-gradient')

        optimizer = dict(type='natural_gradient', learning_rate=1e-3)
        self.unittest(
            optimizer=optimizer,
            policy=dict(network=dict(type='auto', size=8, internal_rnn=False))
            # TODO: shouldn't be necessary!
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

        optimizer = dict(type='synchronization')
        self.unittest(exclude_bounded_action=True, baseline_optimizer=optimizer)  # TODO: shouldn't be necessary!

    def test_tf_optimizer(self):
        self.start_tests(name='tf-optimizer')

        optimizer = dict(type='adam', learning_rate=1e-3)
        self.unittest(optimizer=optimizer)

        optimizer = dict(
            type='radam', learning_rate=1e-3, decoupled_weight_decay=0.01, lookahead=True,
            moving_average=True
        )
        self.unittest(optimizer=optimizer)
