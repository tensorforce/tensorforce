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

import copy
import os
import sys
import unittest

from tensorforce.agents import VPGAgent
from tensorforce.tests.unittest_base import UnittestBase


class TestSaving(UnittestBase, unittest.TestCase):

    agent = VPGAgent
    config = dict(update_mode=dict(batch_size=2))
    directory = 'saving-test'

    def saving_prepare(self, name, **kwargs):
        states = dict(type='float', shape=(1,))

        actions = dict(type='int', shape=(), num_values=3)

        network = [dict(type='dense', size=32), dict(type='dense', size=32)]

        agent, environment = self.prepare(
            name=name, states=states, actions=actions, network=network, **kwargs
        )

        return agent, environment

    def test_explicit_default(self):
        agent, environment = self.saving_prepare(name='explicit-default')

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save_model(directory=self.__class__.directory, filename=None)

        agent.close()

        restored_agent.restore_model(directory=self.__class__.directory, filename=None)

        actions = restored_agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        restored_agent.observe(terminal=terminal, reward=reward)

        restored_agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.meta'))
        os.rmdir(path=self.__class__.directory)

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_explicit_filename(self):
        agent, environment = self.saving_prepare(name='explicit-filename')

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save_model(directory=self.__class__.directory, filename='test')

        agent.close()

        restored_agent.restore_model(directory=self.__class__.directory, filename='test-1')

        actions = restored_agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        restored_agent.observe(terminal=terminal, reward=reward)

        restored_agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'test-1.meta'))
        os.rmdir(path=self.__class__.directory)

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_explicit_no_timestep(self):
        agent, environment = self.saving_prepare(name='explicit-no-timestep')

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save_model(directory=self.__class__.directory, append_timestep=False)

        agent.close()

        restored_agent.restore_model(directory=self.__class__.directory)

        actions = restored_agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        restored_agent.observe(terminal=terminal, reward=reward)

        restored_agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'model.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model.meta'))
        os.rmdir(path=self.__class__.directory)

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_saver_default(self):
        saver = dict(directory=self.__class__.directory)

        agent, environment = self.saving_prepare(name='saver-default', saver=saver)

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()

        restored_agent.initialize()

        actions = restored_agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        restored_agent.observe(terminal=terminal, reward=reward)

        restored_agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_saver_filename(self):
        saver = dict(directory=self.__class__.directory, filename='test')

        agent, environment = self.saving_prepare(name='saver-filename', saver=saver)

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()

        restored_agent.initialize()

        actions = restored_agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        restored_agent.observe(terminal=terminal, reward=reward)

        restored_agent.close()
        environment.close()

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

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_saver_steps(self):
        saver = dict(directory=self.__class__.directory, steps=2)

        agent, environment = self.saving_prepare(name='saver-filename', saver=saver)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-3.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-3.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-3.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_saver_no_load(self):
        saver = dict(directory=self.__class__.directory)

        agent, environment = self.saving_prepare(name='saver-no-load', saver=saver)

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()

        restored_agent.model.saver_spec['load'] = False

        restored_agent.initialize()

        actions = restored_agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        restored_agent.observe(terminal=terminal, reward=reward)

        restored_agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        sys.stdout.flush()
        self.assertTrue(expr=True)

    def test_saver_load_filename(self):
        saver = dict(directory=self.__class__.directory)

        agent, environment = self.saving_prepare(name='saver-load-filename', saver=saver)

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()

        restored_agent.model.saver_spec['load'] = 'model-0'

        restored_agent.initialize()

        actions = restored_agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        restored_agent.observe(terminal=terminal, reward=reward)

        restored_agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'graph.pbtxt'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-0.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.')
            break
        os.rmdir(path=self.__class__.directory)

        sys.stdout.flush()
        self.assertTrue(expr=True)
