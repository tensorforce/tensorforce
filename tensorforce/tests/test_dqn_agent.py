# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest

from tensorforce.tests.base_agent_test import BaseAgentTest
from tensorforce.agents import DQNAgent


class TestDQNAgent(BaseAgentTest, unittest.TestCase):

    agent = DQNAgent
    config = dict(
        update_mode=dict(
            unit='timesteps',
            batch_size=8,
            frequency=8
        ),
        memory=dict(
            type='replay',
            include_next_states=True,
            capacity=100
        ),
        # memory=dict(
        #     type='prioritized_replay',
        #     include_next_states=True,
        #     buffer_size=50,
        #     capacity=100
        # ),
        optimizer=dict(
            type='adam',
            learning_rate=1e-2
        ),
        states_preprocessing=[
            dict(type='running_standardize'),
            dict(type='sequence')
        ],
        target_sync_frequency=10,
        # Comment in to test exploration types
        # actions_exploration_spec=dict(
        #     type="epsilon_decay",
        #     initial_epsilon=1.0,
        #     final_epsilon=0.1,
        #     timesteps=10
        # ),
        # actions_exploration_spec=dict(
        #     type="epsilon_anneal",
        #     initial_epsilon=1.0,
        #     final_epsilon=0.1,
        #     timesteps=10
        # )
    )

    exclude_float = True
    exclude_bounded = True
