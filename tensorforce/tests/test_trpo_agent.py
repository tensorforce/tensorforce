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
from tensorforce.agents import TRPOAgent


class TestTRPOAgent(BaseAgentTest, unittest.TestCase):

    agent = TRPOAgent
    config = dict(
        update_mode=dict(
            unit='episodes',
            batch_size=4,
            frequency=4
        ),
        memory=dict(
            type='latest',
            include_next_states=False,
            capacity=100
        ),
        learning_rate=1e-2
    )

    # multi_config = dict(
    #     batch_size=64,
    #     learning_rate=0.1
    # )
