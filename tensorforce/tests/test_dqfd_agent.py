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

from six import xrange
import unittest

from tensorforce import Configuration
from tensorforce.agents import DQFDAgent
from tensorforce.tests.base_agent_test import BaseAgentTest


class TestDQNAgent(BaseAgentTest, unittest.TestCase):

    agent = DQFDAgent
    deterministic = True
    config = Configuration(
        batch_size=8,
        memory_capacity=800,
        first_update=80,
        target_update_frequency=20,
        demo_memory_capacity=100,
        demo_sampling_ratio=0.2
    )
                # memory=dict(
                #     type='replay',
                #     random_sampling=True
                # ),

    def pre_run(self, agent, environment):
        # First generate demonstration data and pretrain
        demonstrations = list()
        terminal = True

        for n in xrange(50):
            if terminal:
                state = environment.reset()

            actions = dict()

            # TODO: shape parameter, np array

            if 'type' in environment.actions:
                if environment.actions['type'] == 'bool':
                    actions = True
                elif environment.actions['type'] == 'int':
                    actions = 1
                elif environment.actions['type'] == 'float':
                    actions = 1.0

            else:
                for name, action in environment.actions.items():
                    if environment.actions['type'] == 'bool':
                        actions[name] = True
                    elif environment.actions['type'] == 'int':
                        actions[name] = 1
                    elif environment.actions['type'] == 'float':
                        actions[name] = 1.0

            state, terminal, reward = environment.execute(action=action)

            demonstration = dict(states=state, internal=[], actions=action, terminal=terminal, reward=reward)
            demonstrations.append(demonstration)

        agent.import_demonstrations(demonstrations)
        agent.pretrain(steps=1000)
