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

import numpy as np

from tensorforce import Agent, Environment
from test.unittest_base import UnittestBase


class TestSaving(UnittestBase, unittest.TestCase):

    min_timesteps = 3
    require_observe = True

    directory = 'test/test-saving'

    def test_config(self):
        # FEATURES.MD
        self.start_tests(name='config')

        # Remove directory if exists
        if os.path.exists(path=self.__class__.directory):
            for filename in os.listdir(path=self.__class__.directory):
                os.remove(path=os.path.join(self.__class__.directory, filename))
            os.rmdir(path=self.__class__.directory)

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
        # FEATURES.MD
        self.start_tests(name='explicit')

        # Remove directory if exists
        if os.path.exists(path=self.__class__.directory):
            for filename in os.listdir(path=self.__class__.directory):
                os.remove(path=os.path.join(self.__class__.directory, filename))
            os.rmdir(path=self.__class__.directory)

        agent, environment = self.prepare(memory=50, update=dict(unit='episodes', batch_size=1))
        states = environment.reset()

        # save: default tensorflow format
        weights0 = agent.get_variable(variable='policy/policy-network/dense0/weights')
        agent.save(directory=self.__class__.directory)
        agent.close()
        self.finished_test()

        # load: only directory
        agent = Agent.load(directory=self.__class__.directory, environment=environment)
        x = agent.get_variable(variable='policy/policy-network/dense0/weights')
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 0)
        self.finished_test()

        # one timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        # save: numpy format, append timesteps
        weights1 = agent.get_variable(variable='policy/policy-network/dense0/weights')
        agent.save(directory=self.__class__.directory, format='numpy', append='timesteps')
        agent.close()
        self.finished_test()

        # load: numpy format and directory
        agent = Agent.load(
            directory=self.__class__.directory,  format='numpy', environment=environment
        )
        x = agent.get_variable(variable='policy/policy-network/dense0/weights')
        self.assertTrue((x == weights1).all())
        self.assertEqual(agent.timesteps, 1)
        self.finished_test()

        # one timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        # save: numpy format, append timesteps
        weights2 = agent.get_variable(variable='policy/policy-network/dense0/weights')
        agent.save(directory=self.__class__.directory, format='numpy', append='timesteps')
        agent.close()
        self.finished_test()

        # load: numpy format and directory
        agent = Agent.load(
            directory=self.__class__.directory, format='numpy', environment=environment
        )
        x = agent.get_variable(variable='policy/policy-network/dense0/weights')
        self.assertTrue((x == weights2).all())
        self.assertEqual(agent.timesteps, 2)
        self.finished_test()

        # one episode
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

        # save: hdf5 format, filename, append episodes
        weights3 = agent.get_variable(variable='policy/policy-network/dense0/weights')
        self.assertFalse(not (weights3 == weights2).all())
        agent.save(
            directory=self.__class__.directory, filename='agent2', format='hdf5', append='episodes'
        )
        agent.close()
        self.finished_test()

        # env close
        environment.close()

        # differing agent config: episode length, update, parallel_interactions
        environment = Environment.create(environment=self.environment_spec(max_episode_timesteps=7))

        # load: filename (hdf5 format implicit)
        agent = Agent.load(
            directory=self.__class__.directory, filename='agent2', environment=environment,
            update=dict(unit='episodes', batch_size=2), parallel_interactions=2
        )
        x = agent.get_variable(variable='policy/policy-network/dense0/weights')
        self.assertTrue((x == weights3).all())
        self.assertEqual(agent.episodes, 1)
        agent.close()
        self.finished_test()

        # load: tensorflow format (filename explicit)
        agent = Agent.load(
            directory=self.__class__.directory, format='tensorflow', environment=environment,
            update=dict(unit='episodes', batch_size=2), parallel_interactions=2
        )
        x = agent.get_variable(variable='policy/policy-network/dense0/weights')
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 0)
        self.assertEqual(agent.episodes, 0)
        agent.close()
        self.finished_test()

        # load: numpy format, full filename including timesteps suffix
        agent = Agent.load(
            directory=self.__class__.directory, filename='agent-1', format='numpy',
            environment=environment, update=dict(unit='episodes', batch_size=2),
            parallel_interactions=2
        )
        x = agent.get_variable(variable='policy/policy-network/dense0/weights')
        self.assertTrue((x == weights1).all())
        self.assertEqual(agent.timesteps, 1)
        self.assertEqual(agent.episodes, 0)
        agent.close()
        self.finished_test()

        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.npz'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.npz'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent2.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent2-1.hdf5'))
        os.rmdir(path=self.__class__.directory)

        self.finished_test()
