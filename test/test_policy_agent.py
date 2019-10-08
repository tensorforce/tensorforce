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

from tensorforce import util
from test.unittest_agent import UnittestAgent


class TestPolicyAgent(UnittestAgent, unittest.TestCase):

    def test_act_experience_update(self):
        self.start_tests(name='act-experience-update')

        agent, environment = self.prepare(
            require_all=True, update=dict(unit='episodes', batch_size=1),
            network=dict(type='auto', size=8, internal_rnn=False)  # TODO: shouldn't be necessary!
        )
        agent.initialize()

        for n in range(2):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states, independent=True)
                next_states, terminal, reward = environment.execute(actions=actions)
                states = util.fmap(function=(lambda x: [x]), xs=states, depth=1)
                actions = util.fmap(function=(lambda x: [x]), xs=actions, depth=1)
                agent.experience(
                    states=states, actions=actions, terminal=[terminal], reward=[reward]
                )
                states = next_states
            agent.update()

        self.finished_test()
