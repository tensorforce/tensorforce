# Copyright 2020 Tensorforce Team. All Rights Reserved.
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
import unittest

import tensorflow as tf

from tensorforce import Agent, Environment, Runner
from test.unittest_base import UnittestBase


class TestSaving(UnittestBase, unittest.TestCase):

    directory = 'test/test-saving'

    def setUp(self):
        super().setUp()
        tf.config.experimental_run_functions_eagerly(run_eagerly=False)

    def test_modules(self):
        self.start_tests(name='modules')

        # Remove directory if exists
        if os.path.exists(path=self.__class__.directory):
            for filename in os.listdir(path=self.__class__.directory):
                os.remove(path=os.path.join(self.__class__.directory, filename))
            os.rmdir(path=self.__class__.directory)

        agent, environment = self.prepare(config=dict(eager_mode=False))
        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        weights0 = agent.model.policy.network.layers[1].weights.numpy()
        # TODO: implement proper Agent name-module iteration
        for module in agent.model.this_submodules:
            # (Model excluded, other submodules recursively included)
            path = module.save(directory=self.__class__.directory)
            assert path == os.path.join(self.__class__.directory, module.name)
        agent.close()
        environment.close()

        agent, environment = self.prepare(config=dict(eager_mode=False))
        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        for module in agent.model.this_submodules:
            module.restore(directory=self.__class__.directory)
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights0).all())
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        for module in agent.model.this_submodules:
            os.remove(path=os.path.join(self.__class__.directory, module.name + '.index'))
            os.remove(path=os.path.join(
                self.__class__.directory, module.name + '.data-00000-of-00001'
            ))
        os.rmdir(path=self.__class__.directory)

        agent.close()
        environment.close()

        self.finished_test()

    def test_explicit(self):
        # FEATURES.MD
        self.start_tests(name='explicit')

        # Remove directory if exists
        if os.path.exists(path=self.__class__.directory):
            for filename in os.listdir(path=self.__class__.directory):
                os.remove(path=os.path.join(self.__class__.directory, filename))
            os.rmdir(path=self.__class__.directory)

        policy = dict(network=dict(type='auto', size=8, depth=1, rnn=False))
        update = dict(unit='episodes', batch_size=1)
        # TODO: no
        agent, environment = self.prepare(
            policy=policy, memory=50, update=update, config=dict(eager_mode=False)
        )
        states = environment.reset()

        # save: default checkpoint format
        weights0 = agent.model.policy.network.layers[1].weights.numpy()
        agent.save(directory=self.__class__.directory)
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        self.assertEqual(agent.timesteps, 1)
        agent.close()
        self.finished_test()

        # load: only directory
        agent = Agent.load(directory=self.__class__.directory, environment=environment)
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 0)
        self.finished_test()

        # one timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        # save: numpy format, append timesteps
        agent.save(directory=self.__class__.directory, format='numpy', append='timesteps')
        agent.close()
        self.finished_test()

        # load: numpy format and directory
        agent = Agent.load(
            directory=self.__class__.directory, format='numpy', environment=environment
        )
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 1)
        self.finished_test()

        # one timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        # save: numpy format, append timesteps
        agent.save(directory=self.__class__.directory, format='numpy', append='timesteps')
        agent.close()
        self.finished_test()

        # load: numpy format and directory
        agent = Agent.load(
            directory=self.__class__.directory, format='numpy', environment=environment
        )
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 2)
        self.finished_test()

        # one episode
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

        # save: hdf5 format, filename, append episodes
        weights1 = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((weights1 != weights0).any())
        self.assertEqual(agent.episodes, 1)
        agent.save(
            directory=self.__class__.directory, filename='agent2', format='hdf5', append='episodes'
        )
        agent.close()
        self.finished_test()

        # env close
        environment.close()

        # differing agent config: update, parallel_interactions
        # TODO: episode length, others?
        environment = Environment.create(environment=self.environment_spec())

        # load: filename (hdf5 format implicit)
        update['batch_size'] = 2
        agent = Agent.load(
            directory=self.__class__.directory, filename='agent2', environment=environment,
            policy=policy, update=update, parallel_interactions=2
        )
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights1).all())
        self.assertEqual(agent.episodes, 1)
        agent.close()
        self.finished_test()

        # load: tensorflow format (filename explicit)
        # TODO: parallel_interactions=2 should be possible, but problematic if all variables are
        # saved in checkpoint format
        agent = Agent.load(
            directory=self.__class__.directory, format='checkpoint', environment=environment,
            policy=policy, update=update, parallel_interactions=1
        )
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 0)
        self.assertEqual(agent.episodes, 0)
        agent.close()
        self.finished_test()

        # load: numpy format, full filename including timesteps suffix
        agent = Agent.load(
            directory=self.__class__.directory, filename='agent-1', format='numpy',
            environment=environment, policy=policy, update=update, parallel_interactions=2
        )
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 1)
        self.assertEqual(agent.episodes, 0)
        self.finished_test()

        # three episodes (due to batch_size change, hence mismatch with loaded internal last_update)
        for _ in range(3):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
        self.assertEqual(agent.updates, 1)

        # save: saved-model format, append updates
        agent.save(directory=self.__class__.directory, format='saved-model', append='updates')
        agent.close()

        # load: saved-model format
        agent = tf.saved_model.load(export_dir=os.path.join(self.__class__.directory, 'agent-1'))
        act = next(iter(agent._independent_act_graphs.values()))

        # one episode
        states = environment.reset()
        terminal = False
        while not terminal:
            # Turn dicts into lists and batch inputs
            auxiliaries = [[[states.pop('int_action_mask')]]]
            states = [[state] for state in states.values()]
            actions = act(states, auxiliaries)
            # Split result dict and unbatch values
            actions = {
                name: value.numpy().item() if value.shape == (1,) else value.numpy()[0]
                for name, value in actions.items()
            }
            states, terminal, _ = environment.execute(actions=actions)

        # agent.close()
        environment.close()

        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.npz'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-2.npz'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent2.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent2-1.hdf5'))
        os.rmdir(path=os.path.join(self.__class__.directory, 'agent-1', 'assets'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1', 'variables', 'variables.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1', 'variables', 'variables.index'))
        os.rmdir(path=os.path.join(self.__class__.directory, 'agent-1', 'variables'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1', 'saved_model.pb'))
        os.rmdir(path=os.path.join(self.__class__.directory, 'agent-1'))
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

    def test_config(self):
        # FEATURES.MD
        self.start_tests(name='config')

        # Remove directory if exists
        if os.path.exists(path=self.__class__.directory):
            for filename in os.listdir(path=self.__class__.directory):
                os.remove(path=os.path.join(self.__class__.directory, filename))
            os.rmdir(path=self.__class__.directory)

        # save: before first timestep
        update = dict(unit='episodes', batch_size=1)
        saver = dict(directory=self.__class__.directory, frequency=1)
        agent, environment = self.prepare(update=update, saver=saver, config=dict(eager_mode=False))
        weights0 = agent.model.policy.network.layers[1].weights.numpy()
        states = environment.reset()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        updated = agent.observe(terminal=terminal, reward=reward)
        agent.close()
        self.finished_test()

        # load: from given directory
        agent = Agent.load(directory=self.__class__.directory, environment=environment)
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights0).all())
        self.assertEqual(agent.timesteps, 0)
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            updated = agent.observe(terminal=terminal, reward=reward)
        self.assertTrue(updated)
        weights1 = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((weights1 != weights0).any())
        timesteps = agent.timesteps
        agent.close()
        self.finished_test()

        # load: from given directory
        agent = Agent.load(directory=self.__class__.directory, environment=environment)
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights1).all())
        self.assertEqual(agent.timesteps, timesteps)
        agent.close()
        environment.close()
        self.finished_test()

        # create, not load
        agent, environment = self.prepare(update=update, saver=saver, config=dict(eager_mode=False))
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x != weights0).any())
        self.assertTrue((x != weights1).any())
        self.assertEqual(agent.timesteps, 0)
        states = environment.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            updated = agent.observe(terminal=terminal, reward=reward)
        self.assertTrue(updated)
        weights2 = agent.model.policy.network.layers[1].weights.numpy()
        agent.close()
        self.finished_test()

        # load: from given directory
        agent = Agent.load(directory=self.__class__.directory, environment=environment)
        x = agent.model.policy.network.layers[1].weights.numpy()
        self.assertTrue((x == weights2).all())
        agent.close()
        environment.close()
        self.finished_test()

        os.remove(path=os.path.join(self.__class__.directory, 'agent.json'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-0.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.data-00000-of-00001'))
        os.remove(path=os.path.join(self.__class__.directory, 'agent-1.index'))
        os.remove(path=os.path.join(self.__class__.directory, 'checkpoint'))
        os.rmdir(path=self.__class__.directory)

        self.finished_test()

    def test_load_performance(self):
        self.start_tests(name='load-performance')

        environment = Environment.create(environment='CartPole-v1')

        agent = Agent.load(
            directory='test/data', filename='ppo-checkpoint', format='checkpoint',
            environment=environment
        )
        runner = Runner(agent=agent, environment=environment, evaluation=True)
        runner.run(num_episodes=10, use_tqdm=False)
        self.assertTrue(all(episode_reward == 500 for episode_reward in runner.episode_rewards))
        runner.close()
        agent.close()
        self.finished_test()

        agent = Agent.load(
            directory='test/data', filename='ppo-checkpoint', format='numpy',
            environment=environment
        )
        runner = Runner(agent=agent, environment=environment, evaluation=True)
        runner.run(num_episodes=10, use_tqdm=False)
        self.assertTrue(all(episode_reward == 500 for episode_reward in runner.episode_rewards))
        runner.close()
        agent.close()
        self.finished_test()

        agent = Agent.load(
            directory='test/data', filename='ppo-checkpoint', format='hdf5',
            environment=environment
        )
        runner = Runner(agent=agent, environment=environment, evaluation=True)
        runner.run(num_episodes=10, use_tqdm=False)
        self.assertTrue(all(episode_reward == 500 for episode_reward in runner.episode_rewards))
        runner.close()
        agent.close()
        self.finished_test()
