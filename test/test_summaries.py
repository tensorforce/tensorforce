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

        learning_rate = dict(
            type='linear', unit='updates', num_steps=10, initial_value=1e-3, final_value=1e-4
        )
        horizon = dict(type='linear', unit='episodes', num_steps=2, initial_value=2, final_value=4)
        exploration = dict(
            type='exponential', unit='timesteps', num_steps=5, initial_value=0.1, decay_rate=0.5
        )

        with TemporaryDirectory() as directory:
            agent, environment = self.prepare(
                optimizer=dict(optimizer='adam', learning_rate=learning_rate),
                reward_estimation=dict(
                    horizon=horizon, return_processing=dict(type='clipping', lower=-1.0, upper=1.0)
                ), exploration=exploration,
                config=dict(eager_mode=False, create_debug_assertions=True, tf_log_level=20),
                summarizer=dict(directory=directory, summaries='all')
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
