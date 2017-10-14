
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

from tensorforce import Configuration
from tensorforce.agents import ConstantAgent
from tensorforce.tests.base_agent_test import BaseAgentTest


class TestConstantAgent(BaseAgentTest, unittest.TestCase):

    #TODO we expect this to fail, need flag to indicate this in base tests
    agent = ConstantAgent
    deterministic = False
    config = Configuration()

