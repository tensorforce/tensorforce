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

from test.unittest_base import UnittestBase


class TestMemories(UnittestBase, unittest.TestCase):

    num_episodes = 2

    def test_recent(self):
        self.start_tests(name='recent')

        memory = dict(type='recent', capacity=9)
        update = dict(unit='timesteps', batch_size=4)
        self.unittest(update=update, memory=memory)

        memory = dict(type='recent', capacity=10)
        update = dict(unit='episodes', batch_size=1)
        self.unittest(update=update, memory=memory)

    def test_replay(self):
        self.start_tests(name='replay')

        memory = dict(type='replay', capacity=9)
        update = dict(unit='timesteps', batch_size=4)
        self.unittest(update=update, memory=memory)

        memory = dict(type='replay', capacity=10)
        update = dict(unit='episodes', batch_size=1)
        self.unittest(update=update, memory=memory)

        memory = 9
        update = 4
        self.unittest(update=update, memory=memory)
