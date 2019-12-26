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

import os
import unittest

from test.unittest_agent import UnittestAgent


class TestTensorforceAgent(UnittestAgent, unittest.TestCase):

    directory = 'test-recording'

    def test_act_experience_update(self):
        self.start_tests(name='act-experience-update')

        states = dict(type='float', shape=(1,))
        actions = dict(type=self.__class__.replacement_action, shape=())

        agent, environment = self.prepare(
            states=states, actions=actions, require_all=True,
            update=dict(unit='episodes', batch_size=1),
            policy=dict(network=dict(type='auto', size=8, internal_rnn=False))
            # TODO: shouldn't be necessary!
        )

        for n in range(2):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states, independent=True)
                next_states, terminal, reward = environment.execute(actions=actions)
                agent.experience(states=states, actions=actions, terminal=terminal, reward=reward)
                states = next_states
            agent.update()

        self.finished_test()

    def test_pretrain(self):
        # FEATURES.MD
        self.start_tests(name='pretrain')

        states = dict(type='float', shape=(1,))
        actions = dict(type=self.__class__.replacement_action, shape=())

        agent, environment = self.prepare(
            states=states, actions=actions, require_all=True, buffer_observe=False, update=1,
            policy=dict(network=dict(type='auto', size=8, internal_rnn=False)),
            # TODO: shouldn't be necessary!
            recorder=dict(directory=self.__class__.directory)
        )

        for _ in range(3):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

        agent.pretrain(
            directory=self.__class__.directory, num_iterations=2, num_traces=2, num_updates=3
        )

        agent.close()
        environment.close()

        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('trace-')
        os.rmdir(path=self.__class__.directory)

        self.finished_test()
