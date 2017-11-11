
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

from tensorforce.agents import ConstantAgent
from tensorforce.tests.base_agent_test import BaseAgentTest


class TestConstantAgent(BaseAgentTest, unittest.TestCase):

    agent = ConstantAgent
    deterministic = False
    requires_network = False

    # Just testing one test, otherwise we would have to specify constant values of every type for every
    # test and override all base tests
    kwargs = dict(
        action_values=dict(
            action=1.0
        )
    )

    exclude_bool = True
    exclude_int = True
    exclude_bounded = True
    exclude_multi = True
    exclude_lstm = True
