# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import pytest
import time
import unittest

from tensorforce.agents import Agent
from test.unittest_base import UnittestBase


class TestSaving(UnittestBase, unittest.TestCase):

    min_timesteps = 3
    require_observe = True

    directory = 'test-saving'

    def test_config(self):
        # FEATURES.MD
        self.start_tests(name='config')

        # default
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(saver=saver)

        states = environment.reset()
        agent.close()

        agent = Agent.load(directory=self.__class__.directory, environment=environment)

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

        # single then parallel and different episode length
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(
            memory=50, update=dict(unit='episodes', batch_size=1), saver=saver
        )

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        agent, environment = self.prepare(
            update=dict(unit='episodes', batch_size=1), saver=saver, max_episode_timesteps=7,
            parallel_interactions=2
        )

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

        # no load
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(saver=saver)

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        saver = dict(directory=self.__class__.directory, load=False)
        agent, environment = self.prepare(saver=saver)

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

    @pytest.mark.skip(reason='currently takes too long')
    def test_config_extended(self):
        self.start_tests(name='config extended')

        # filename
        saver = dict(directory=self.__class__.directory, filename='test')
        agent, environment = self.prepare(saver=saver)

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()

        agent = Agent.load(
            directory=self.__class__.directory, filename='test', environment=environment
        )

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'test.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-2.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-2.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-2.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

        # frequency
        saver = dict(directory=self.__class__.directory, frequency=1)
        agent, environment = self.prepare(saver=saver)

        states = environment.reset()
        time.sleep(1)
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        time.sleep(1)
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.'), filename
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

        # load filename
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(saver=saver)

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        saver = dict(directory=self.__class__.directory, load='agent-0')
        agent, environment = self.prepare(saver=saver)

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

    def test_explicit(self):
        self.start_tests(name='explicit')

        # default
        agent, environment = self.prepare()

        states = environment.reset()
        agent.save(directory=self.__class__.directory)
        agent.close()

        agent = Agent.load(directory=self.__class__.directory, environment=environment)

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.meta'))
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

        # single then parallel and different episode length
        agent, environment = self.prepare(memory=50, update=dict(unit='episodes', batch_size=1))

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save(directory=self.__class__.directory)
        agent.close()
        environment.close()

        agent, environment = self.prepare(
            update=dict(unit='episodes', batch_size=1), max_episode_timesteps=7,
            parallel_interactions=2
        )

        agent.restore(directory=self.__class__.directory)

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.meta'))
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

    @pytest.mark.skip(reason='currently takes too long')
    def test_explicit_extended(self):
        self.start_tests(name='explicit extended')

        # filename
        agent, environment = self.prepare()

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save(directory=self.__class__.directory, filename='test')
        agent.close()

        agent = Agent.load(
            directory=self.__class__.directory, filename='test', environment=environment
        )

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'test.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.meta'))
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

        # no timestep
        agent, environment = self.prepare()

        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save(directory=self.__class__.directory, append_timestep=False)
        agent.close()

        agent = Agent.load(directory=self.__class__.directory, environment=environment)

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent.meta'))
        os.rmdir(path=self.__class__.directory)

        self.finished_test()
