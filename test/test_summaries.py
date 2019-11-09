# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import unittest

from test.unittest_base import UnittestBase


class TestSummaries(UnittestBase, unittest.TestCase):

    exclude_bounded_action = True  # TODO: shouldn't be necessary!
    require_observe = True

    directory = 'test-summaries'

    def test_summaries(self):
        # FEATURES.MD
        self.start_tests()

        # TODO: 'dropout'
        reward_estimation = dict(horizon=2, estimate_horizon='late')
        baseline_policy = dict(network=dict(type='auto', size=8, internal_rnn=1))
        baseline_objective = 'policy_gradient'
        baseline_optimizer = 'adam'

        self.unittest(
            summarizer=dict(directory=self.__class__.directory, labels='all', frequency=2),
            reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        for directory in os.listdir(path=self.__class__.directory):
            directory = os.path.join(self.__class__.directory, directory)
            for filename in os.listdir(path=directory):
                os.remove(path=os.path.join(directory, filename))
                assert filename.startswith('events.out.tfevents.')
                break
            os.rmdir(path=directory)
        os.rmdir(path=self.__class__.directory)

        self.finished_test()
