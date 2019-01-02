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

from tensorforce.agents import DDPGAgent
from tensorforce.tests.agent_unittest import AgentUnittest


class TestDDPGAgent(AgentUnittest, unittest.TestCase):

    agent = DDPGAgent

    config = dict(
        critic_network=[
            [
                dict(type='input', names=['state']),
                dict(type='global_pooling'),
                dict(type='dense', size=32),
                dict(type='output', name='state_output')
            ],
            [
                dict(type='input', names=['state_output', 'action'], aggregation_type='concat'),
                dict(type='dense', size=32),
                dict(type='dense', size=1)
            ]
        ]
    )
