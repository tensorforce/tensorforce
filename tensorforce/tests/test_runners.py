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

import copy
import sys
import unittest

from tensorforce.agents import VPGAgent
from tensorforce.execution import ParallelRunner
from tensorforce.tests.unittest_base import UnittestBase


class TestRunners(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))

    def test_parallel_runner(self):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        agent, environment1 = self.prepare(
            name='parallel-runner', states=states, actions=actions, network=network,
            parallel_interactions=2
        )
        environment2 = copy.deepcopy(environment1)

        runner = ParallelRunner(agent=agent, environments=[environment1, environment2])
        runner.run(num_episodes=10)
        runner.close()

        sys.stdout.flush()
        self.assertTrue(expr=True)
