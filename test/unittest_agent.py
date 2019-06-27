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

from test.unittest_base import UnittestBase


class UnittestAgent(UnittestBase):
    """
    Collection of unit-tests for agent functionality.
    """

    replacement_action = 'bool'
    action_masks = True
    has_experience = True
    has_update = True

    def test_single_state_action(self):
        self.start_tests(name='single-state-action')

        states = dict(type='bool', shape=())
        if self.__class__.exclude_bool_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='bool', shape=())
        self.unittest(states=states, actions=actions, action_masks=self.__class__.action_masks)

        states = dict(type='int', shape=(), num_values=3)
        if self.__class__.exclude_int_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='int', shape=(), num_values=3)
        self.unittest(states=states, actions=actions, action_masks=self.__class__.action_masks)

        states = dict(type='float', shape=(1,))
        if self.__class__.exclude_float_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='float', shape=())
        self.unittest(states=states, actions=actions, action_masks=self.__class__.action_masks)

        states = dict(type='float', shape=(1,), min_value=-1.0, max_value=1.0)
        if self.__class__.exclude_bounded_action:
            actions = dict(type=self.__class__.replacement_action, shape=())
        else:
            actions = dict(type='float', shape=(), min_value=-1.0, max_value=1.0)
        self.unittest(states=states, actions=actions, action_masks=self.__class__.action_masks)

    def test_full(self):
        self.start_tests(name='full')
        self.unittest(action_masks=self.__class__.action_masks)

    def test_pretrain(self):
        if not self.__class__.has_experience or not self.__class__.has_update:
            return

        self.start_tests(name='pretrain')

        states = dict(type='float', shape=(1,))
        actions = dict(type=self.__class__.replacement_action, shape=())

        agent, environment = self.prepare(
            states=states, actions=actions, action_masks=self.__class__.action_masks,
            buffer_observe=False, update=1,
            network=dict(type='auto', size=8, internal_rnn=False)  # TODO: shouldn't be necessary!
        )

        agent.initialize()

        states_batch = list()
        actions_batch = list()
        terminal_batch = list()
        reward_batch = list()

        states = environment.reset()
        terminal = False
        while not terminal:
            states_batch.append(states)
            actions = agent.act(states=states, independent=True)
            actions_batch.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            terminal_batch.append(terminal)
            reward_batch.append(reward)

        agent.experience(
            states=states_batch, actions=actions_batch, terminal=terminal_batch,
            reward=reward_batch
        )

        agent.update()
        agent.update()
        agent.update()

        agent.close()
        environment.close()

        self.finished_test()

    def test_query(self):
        self.start_tests(name='query')

        states = dict(type='float', shape=(1,))
        actions = dict(type=self.__class__.replacement_action, shape=())

        agent, environment = self.prepare(
            states=states, actions=actions, action_masks=self.__class__.action_masks,
            buffer_observe=False, update=1,
            network=dict(type='auto', size=8, internal_rnn=False)  # TODO: shouldn't be necessary!
        )

        agent.initialize()
        states = environment.reset()

        query = agent.get_query_tensors(function='act')
        actions, queried = agent.act(states=states, query=query)
        self.assertEqual(first=len(queried), second=len(query))

        states, _, reward = environment.execute(actions=actions)

        query = agent.get_query_tensors(function='observe')
        queried = agent.observe(terminal=True, reward=reward, query=query)
        self.assertEqual(first=len(queried), second=len(query))

        states_batch = list()
        actions_batch = list()
        terminal_batch = list()
        reward_batch = list()

        states = environment.reset()
        terminal = False
        while not terminal:
            states_batch.append(states)
            actions = agent.act(states=states, independent=True)
            actions_batch.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            terminal_batch.append(terminal)
            reward_batch.append(reward)

        if self.__class__.has_experience:
            query = agent.get_query_tensors(function='experience')
            queried = agent.experience(
                states=states_batch, actions=actions_batch, terminal=terminal_batch,
                reward=reward_batch, query=query
            )
            self.assertEqual(first=len(queried), second=len(query))

        if self.__class__.has_update:
            query = agent.get_query_tensors(function='update')
            queried = agent.update(query=query)
            self.assertEqual(first=len(queried), second=len(query))

        agent.close()
        environment.close()

        self.finished_test()
