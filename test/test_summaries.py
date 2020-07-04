# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import tensorflow as tf

from test.unittest_base import UnittestBase


class TestSummaries(UnittestBase, unittest.TestCase):

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

        horizon = dict(type='linear', unit='updates', num_steps=2, initial_value=2,  final_value=4)
        baseline_policy = dict(network=dict(type='auto', size=8, depth=1, rnn=1))
        baseline_objective = 'value'
        baseline_optimizer = 'adam'
        preprocessing = dict(reward=dict(type='clipping', upper=0.25))
        exploration = dict(
            type='exponential', unit='episodes', num_steps=3, initial_value=2.0, decay_rate=0.5
        )

        agent, environment = self.prepare(
            summarizer=dict(directory=self.__class__.directory, labels='all'),
            reward_estimation=dict(horizon=horizon), baseline_policy=baseline_policy,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer,
            preprocessing=preprocessing, exploration=exploration,
            config=dict(create_tf_assertions=False, eager_mode=False)
        )

        updates = 0
        episodes = 0
        while episodes < 3 or updates < 3:
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                updates += int(agent.observe(terminal=terminal, reward=reward))
            episodes += 1

        agent.close()
        environment.close()

        for directory in os.listdir(path=self.__class__.directory):
            directory = os.path.join(self.__class__.directory, directory)
            for filename in os.listdir(path=directory):
                os.remove(path=os.path.join(directory, filename))
                assert filename.startswith('events.out.tfevents.')
                break
            os.rmdir(path=directory)
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()
