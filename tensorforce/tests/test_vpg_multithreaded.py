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

import copy
import logging
import sys
import unittest

from tensorforce.agents import VPGAgent
from .minimal_test import MinimalTest
from tensorforce.execution import ThreadedRunner
from tensorforce.execution.threaded_runner import clone_worker_agent


logging.getLogger('tensorflow').disabled = True


class TestVPGMultithreaded(unittest.TestCase):

    def test_multithreaded(self):
        sys.stdout.write('\nVPGAgent (multithreaded):')
        sys.stdout.flush()

        environment = MinimalTest(specification={'int': ()})

        network = [
            dict(type='dense', size=32),
            dict(type='dense', size=32)
        ]
        kwargs = dict(
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
            optimizer=dict(
                type='adam',
                learning_rate=1e-2
            ),
            batched_observe=False
        )
        agent = VPGAgent(
            states=environment.states,
            actions=environment.actions,
            network=network,
            **kwargs
        )

        agents = clone_worker_agent(agent, 5, environment, network, kwargs)
        environments = [environment] + [copy.deepcopy(environment) for n in range(4)]

        runner = ThreadedRunner(agent=agents, environment=environments)

        runner.run(num_episodes=1000)
        runner.close()

        sys.stdout.write(' ran\n')
        sys.stdout.flush()
