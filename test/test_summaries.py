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
from tempfile import TemporaryDirectory
import unittest

from test.unittest_base import UnittestBase


class TestSummaries(UnittestBase, unittest.TestCase):

    def test_summaries(self):
        # FEATURES.MD
        self.start_tests()

        horizon = dict(type='linear', unit='updates', num_steps=2, initial_value=2, final_value=4)
        baseline_policy = dict(network=dict(type='auto', size=8, depth=1, rnn=2))
        baseline_objective = 'value'
        baseline_optimizer = 'adam'
        preprocessing = dict(reward=dict(type='clipping', upper=0.25))
        exploration = dict(
            type='exponential', unit='episodes', num_steps=3, initial_value=2.0, decay_rate=0.5
        )

        with TemporaryDirectory() as directory:
            agent, environment = self.prepare(
                reward_estimation=dict(horizon=horizon), baseline_policy=baseline_policy,
                baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer,
                preprocessing=preprocessing, exploration=exploration,
                config=dict(create_tf_assertions=False, eager_mode=False),
                summarizer=dict(directory=directory, labels='all')
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

            directories = os.listdir(path=directory)
            self.assertEqual(len(directories), 1)
            files = os.listdir(path=os.path.join(directory, directories[0]))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith('events.out.tfevents.'))

        self.finished_test()
