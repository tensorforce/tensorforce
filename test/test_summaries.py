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

import numpy as np

from test.unittest_base import UnittestBase


class TestSummaries(UnittestBase, unittest.TestCase):

    require_observe = True

    directory = 'test/test-summaries'

    def test_summaries(self):
        # FEATURES.MD
        self.start_tests()

        # Remove directory if exists
        if os.path.exists(path=self.__class__.directory):
            for directory in os.listdir(path=self.__class__.directory):
                directory = os.path.join(self.__class__.directory, directory)
                for filename in os.listdir(path=directory):
                    os.remove(path=os.path.join(directory, filename))
                os.rmdir(path=directory)
            os.rmdir(path=self.__class__.directory)

        # Remove directory if exists
        if os.path.exists(path=self.__class__.directory):
            for filename in os.listdir(path=self.__class__.directory):
                os.remove(path=os.path.join(self.__class__.directory, filename))
            os.rmdir(path=self.__class__.directory)

        # TODO: 'dropout'
        reward_estimation = dict(horizon=2, estimate_horizon='late')
        baseline_policy = dict(network=dict(type='auto', size=8, depth=1, internal_rnn=1))
        baseline_objective = 'policy_gradient'
        baseline_optimizer = 'adam'

        agent, environment = self.prepare(
            summarizer=dict(
                directory=self.__class__.directory, labels='all', frequency=2, custom=dict(
                    audio=dict(type='audio', sample_rate=44100, max_outputs=1),
                    histogram=dict(type='histogram'),
                    image=dict(type='image', max_outputs=1),
                    scalar=dict(type='scalar')
                )
            ), reward_estimation=reward_estimation, baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer
        )

        updated = False
        while not updated:
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                updated = agent.observe(terminal=terminal, reward=reward) or updated

        agent.summarize(summary='image', value=np.zeros(shape=(2, 4, 2, 3)))
        agent.summarize(summary='scalar', value=1.0, step=0)
        agent.summarize(summary='scalar', value=2.0, step=1)
        agent.close()
        environment.close()
        self.finished_test()

        for directory in os.listdir(path=self.__class__.directory):
            directory = os.path.join(self.__class__.directory, directory)
            for filename in os.listdir(path=directory):
                os.remove(path=os.path.join(directory, filename))
                assert filename.startswith('events.out.tfevents.')
                break
            os.rmdir(path=directory)
        os.rmdir(path=self.__class__.directory)

        self.finished_test()
