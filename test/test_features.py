# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import os
from random import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tensorforce import Agent, Environment, Runner
from test.unittest_base import UnittestBase


class TestFeatures(UnittestBase, unittest.TestCase):

    def test_masking(self):
        # FEATURES.MD
        self.start_tests(name='masking')

        agent = Agent.create(agent=self.agent_spec(
            states=dict(type='float', shape=(10,)),
            actions=dict(type='int', shape=(), num_values=3)
        ))

        states = dict(
            state=np.random.random_sample(size=(10,)),  # state (default name: "state")
            action_mask=[True, False, True]  # mask as'[ACTION-NAME]_mask' (default name: "action")
        )
        action = agent.act(states=states)
        assert action != 1
