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
import time
import unittest

from test.unittest_base import UnittestBase


class TestSaving(UnittestBase, unittest.TestCase):

    directory = 'test-saving'

    def test_explicit(self):
        self.start_tests(name='explicit')

        # default
        agent, environment = self.prepare()

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save(directory=self.__class__.directory, filename=None)

        agent.close()

        restored_agent.restore(directory=self.__class__.directory, filename=None)

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

        self.finished_test()

        # parallel
        agent, environment = self.prepare(parallel_interactions=2)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save(directory=self.__class__.directory, filename=None)

        agent.close()

        restored_agent, _ = self.prepare()

        restored_agent.restore(directory=self.__class__.directory, filename=None)

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

        self.finished_test()

        # filename
        agent, environment = self.prepare()

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save(directory=self.__class__.directory, filename='test')

        agent.close()

        restored_agent.restore(directory=self.__class__.directory, filename='test-1')

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

        self.finished_test()

        # no timestep
        agent, environment = self.prepare()

        restored_agent = copy.deepcopy(agent)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.save(directory=self.__class__.directory, append_timestep=False)

        agent.close()

        restored_agent.restore(directory=self.__class__.directory)

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

        self.finished_test()

    def test_config(self):
        self.start_tests(name='config')

        # default
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(saver=saver)

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

        self.finished_test()

        # parallel
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(saver=saver, parallel_interactions=2)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        agent.close()

        restored_agent, _ = self.prepare(saver=saver)

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

        self.finished_test()

        # filename
        saver = dict(directory=self.__class__.directory, filename='test')
        agent, environment = self.prepare(saver=saver)

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

        self.finished_test()

        # steps
        saver = dict(directory=self.__class__.directory, frequency=1)
        agent, environment = self.prepare(saver=saver)

        agent.initialize()
        states = environment.reset()

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        time.sleep(1)

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
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-1.meta'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'model-2.meta'))
        for filename in os.listdir(path=self.__class__.directory):
            os.remove(path=os.path.join(self.__class__.directory, filename))
            assert filename.startswith('events.out.tfevents.'), filename
            break
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

        # no load
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(saver=saver)

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

        self.finished_test()

        # load filename
        saver = dict(directory=self.__class__.directory)
        agent, environment = self.prepare(saver=saver)

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

        self.finished_test()
