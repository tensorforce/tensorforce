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

from six.moves import xrange
import unittest
import numpy as np

from tensorforce import util
from tensorforce.agents import DQFDAgent
from tensorforce.tests.base_agent_test import BaseAgentTest


class TestDQFDAgent(BaseAgentTest, unittest.TestCase):

    agent = DQFDAgent
    deterministic = True

    kwargs = dict(
        memory=dict(
            type='replay',
            capacity=1000
        ),
        batch_size=8,
        first_update=10,
        target_sync_frequency=10,
        demo_memory_capacity=100,
        demo_sampling_ratio=0.2
    )

    exclude_float = True
    exclude_bounded = True

    def pre_run(self, agent, environment):
        demonstrations = list()

        agent.reset()
        internals = agent.current_internals
        terminal = True

        for n in xrange(50):
            if terminal:
                state = environment.reset()

            actions = dict()
            # Create demonstration actions of the right shape.
            if 'type' in environment.actions:
                if environment.actions['type'] == 'bool':
                    actions = np.full(
                        shape=(),
                        fill_value=True,
                        dtype=util.np_dtype(environment.actions['type'])
                    )
                elif environment.actions['type'] == 'int':
                    actions = np.full(
                        shape=(),
                        fill_value=1,
                        dtype=util.np_dtype(environment.actions['type'])
                    )
                elif environment.actions['type'] == 'float':
                    actions = np.full(
                        shape=(),
                        fill_value=1.0,
                        dtype=util.np_dtype(environment.actions['type'])
                    )
            else:
                for name, action in environment.actions.items():
                    if action['type'] == 'bool':
                        actions[name] = np.full(
                            shape=action['shape'],
                            fill_value=True,
                            dtype=util.np_dtype(action['type'])
                        )
                    elif action['type'] == 'int':
                        actions[name] = np.full(
                            shape=action['shape'],
                            fill_value=1,
                            dtype=util.np_dtype(action['type'])
                        )
                    elif action['type'] == 'float':
                        actions[name] = np.full(
                            shape=action['shape'],
                            fill_value=1.0,
                            dtype=util.np_dtype(action['type'])
                        )

            state, terminal, reward = environment.execute(actions=actions)

            demonstration = dict(states=state, internals=internals, actions=actions, terminal=terminal, reward=reward)
            demonstrations.append(demonstration)

        agent.import_demonstrations(demonstrations)
        agent.pretrain(steps=1000)
