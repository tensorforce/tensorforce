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
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tensorflow as tf

from tensorforce import Agent, Environment, Runner
from test.unittest_base import UnittestBase


class TestSaving(UnittestBase, unittest.TestCase):

    def test_modules(self):
        self.start_tests(name='modules')

        with TemporaryDirectory() as directory:
            agent, environment = self.prepare(
                config=dict(eager_mode=False, create_debug_assertions=True, tf_log_level=20)
            )
            states = environment.reset()
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            weights0 = agent.model.policy.network.layers[1].weights.numpy()
            for module in agent.model.tensorforce_submodules:
                path = module.save(directory=directory)
                assert path == os.path.join(directory, module.full_name.replace('/', '.'))
            agent.close()
            environment.close()

            agent, environment = self.prepare(
                config=dict(eager_mode=False, create_debug_assertions=True, tf_log_level=20)
            )
            states = environment.reset()
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            for module in agent.model.tensorforce_submodules:
                module.restore(directory=directory)
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights0))
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

            files = set(os.listdir(path=directory))
            self.assertTrue(len(files), 2 * len(agent.model.this_submodules))
            for module in agent.model.tensorforce_submodules:
                filename = module.full_name.replace('/', '.')
                self.assertTrue(filename + '.index' in files)
                self.assertTrue(filename + '.data-00000-of-00001' in files)

        agent.close()
        environment.close()

        self.finished_test()

    def test_explicit(self):
        # FEATURES.MD
        self.start_tests(name='explicit')

        with TemporaryDirectory() as directory:
            update = dict(unit='episodes', batch_size=1)
            agent, environment = self.prepare(
                memory=50, update=update,
                config=dict(eager_mode=False, create_debug_assertions=True, tf_log_level=20)
            )
            states = environment.reset()

            # save: default checkpoint format
            weights0 = agent.model.policy.network.layers[1].weights.numpy()
            agent.save(directory=directory)
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            self.assertEqual(agent.timesteps, 1)
            agent.close()
            self.finished_test()

            # load: only directory
            agent = Agent.load(directory=directory, environment=environment)
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights0))
            self.assertEqual(agent.timesteps, 0)
            self.finished_test()

            # one timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)

            # save: numpy format, append timesteps
            agent.save(directory=directory, format='numpy', append='timesteps')
            agent.close()
            self.finished_test()

            # load: numpy format and directory
            agent = Agent.load(directory=directory, format='numpy', environment=environment)
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights0))
            self.assertEqual(agent.timesteps, 1)
            self.finished_test()

            # one timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

            # save: numpy format, append timesteps
            agent.save(directory=directory, format='numpy', append='timesteps')
            agent.close()
            self.finished_test()

            # load: numpy format and directory
            agent = Agent.load(directory=directory, format='numpy', environment=environment)
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights0))
            self.assertEqual(agent.timesteps, 2)
            self.finished_test()

            # one episode
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

            # save: hdf5 format, filename, append episodes
            weights1 = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(not np.allclose(weights1, weights0))
            self.assertEqual(agent.episodes, 1)
            agent.save(directory=directory, filename='agent2', format='hdf5', append='episodes')
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
                directory=directory, filename='agent2', environment=environment, update=update,
                parallel_interactions=2
            )
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights1))
            self.assertEqual(agent.episodes, 1)
            agent.close()
            self.finished_test()

            # load: tensorflow format (filename explicit)
            # TODO: parallel_interactions=2 should be possible, but problematic if all variables are
            # saved in checkpoint format
            agent = Agent.load(
                directory=directory, format='checkpoint', environment=environment, update=update,
                parallel_interactions=1
            )
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights0))
            self.assertEqual(agent.timesteps, 0)
            self.assertEqual(agent.episodes, 0)
            agent.close()
            self.finished_test()

            # load: numpy format, full filename including timesteps suffix
            agent = Agent.load(
                directory=directory, filename='agent-1', format='numpy', environment=environment,
                update=update, parallel_interactions=2
            )
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights0))
            self.assertEqual(agent.timesteps, 1)
            self.assertEqual(agent.episodes, 0)
            self.finished_test()

            # three episodes (due to batch_size change, mismatch with loaded internal last_update)
            for _ in range(3):
                states = environment.reset()
                terminal = False
                while not terminal:
                    actions = agent.act(states=states)
                    states, terminal, reward = environment.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)
            self.assertEqual(agent.updates, 1)

            # save: saved-model format, append updates
            agent.save(directory=directory, format='saved-model', append='updates')
            agent.close()

            # saved-model functions
            def batch(x):
                return np.expand_dims(x, axis=0)

            def unbatch(x):
                if isinstance(x, tf.Tensor):
                    x = x.numpy()
                if x.shape == (1,):
                    return x.item()
                else:
                    return np.squeeze(x, axis=0)

            def recursive_map(function, dictionary):
                mapped = dict()
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        mapped[key] = recursive_map(function, value)
                    else:
                        mapped[key] = function(value)
                return mapped

            # load: saved-model format
            agent = tf.saved_model.load(export_dir=os.path.join(directory, 'agent-1'))

            # one episode
            states = environment.reset()
            internals = agent.initial_internals()
            internals = recursive_map(batch, internals)
            terminal = False
            while not terminal:
                auxiliaries = dict(int_action=dict(mask=batch(states.pop('int_action_mask'))))
                states = recursive_map(batch, states)
                actions_internals = agent.act(states, internals, auxiliaries, False)
                actions = actions_internals['actions']
                internals = actions_internals['internals']
                actions = recursive_map(unbatch, actions)
                states, terminal, _ = environment.execute(actions=actions)

            environment.close()

            # saved-model format with singleton state/action, no internals, no masking
            policy = dict(network=dict(type='auto', size=8, depth=1, rnn=False))
            update = dict(unit='episodes', batch_size=1)
            baseline = dict(network=dict(type='auto', size=7, depth=1, rnn=False))
            agent, environment = self.prepare(
                states=dict(type='float', shape=(), min_value=1.0, max_value=2.0),
                actions=dict(type='float', shape=(), min_value=1.0, max_value=2.0),
                policy=policy, update=update, baseline=baseline,
                config=dict(eager_mode=False, create_debug_assertions=True, tf_log_level=20)
            )

            # one episode
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
            self.assertEqual(agent.updates, 1)

            # save: saved-model format, append updates
            agent.save(directory=directory, format='saved-model', append='updates')
            agent.close()

            # load: saved-model format
            agent = tf.saved_model.load(export_dir=os.path.join(directory, 'agent-1'))

            # one episode
            states = environment.reset()
            terminal = False
            while not terminal:
                states = batch(states)
                actions = agent.act(states, True)
                actions = unbatch(actions)
                states, terminal, _ = environment.execute(actions=actions)

            environment.close()

            files = set(os.listdir(path=directory))
            self.assertTrue(files == {
                'agent.json', 'agent-1', 'agent-1.data-00000-of-00001', 'agent-1.index',
                'agent-1.npz', 'agent2.json', 'agent-2.npz', 'agent2-1.hdf5', 'checkpoint'
            })
            files = set(os.listdir(path=os.path.join(directory, 'agent-1')))
            self.assertTrue(files == {'assets', 'saved_model.pb', 'variables'})
            files = set(os.listdir(path=os.path.join(directory, 'agent-1', 'variables')))
            self.assertTrue(files == {'variables.data-00000-of-00001', 'variables.index'})

        self.finished_test()

    def test_config(self):
        # FEATURES.MD
        self.start_tests(name='config')

        with TemporaryDirectory() as directory:
            # save: before first timestep
            update = dict(unit='episodes', batch_size=1)
            saver = dict(directory=directory, frequency=1)
            agent, environment = self.prepare(
                update=update, saver=saver,
                config=dict(eager_mode=False, create_debug_assertions=True, tf_log_level=20)
            )
            weights0 = agent.model.policy.network.layers[1].weights.numpy()
            states = environment.reset()
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            updated = agent.observe(terminal=terminal, reward=reward)
            agent.close()
            self.finished_test()

            # load: from given directory
            agent = Agent.load(directory=directory, environment=environment)
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights0))
            self.assertEqual(agent.timesteps, 0)
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                updated = agent.observe(terminal=terminal, reward=reward)
            self.assertTrue(updated)
            weights1 = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(not np.allclose(weights1, weights0))
            timesteps = agent.timesteps
            agent.close()
            self.finished_test()

            # load: from given directory
            agent = Agent.load(directory=directory, environment=environment)
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights1))
            self.assertEqual(agent.timesteps, timesteps)
            agent.close()
            environment.close()
            self.finished_test()

            # create, not load
            agent, environment = self.prepare(
                update=update, saver=saver,
                config=dict(eager_mode=False, create_debug_assertions=True, tf_log_level=20)
            )
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(not np.allclose(x, weights0))
            self.assertTrue(not np.allclose(x, weights1))
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
            agent = Agent.load(directory=directory, environment=environment)
            x = agent.model.policy.network.layers[1].weights.numpy()
            self.assertTrue(np.allclose(x, weights2))
            agent.close()
            environment.close()
            self.finished_test()

            files = set(os.listdir(path=directory))
            self.assertTrue(files == {
                'agent.json', 'agent-0.data-00000-of-00001', 'agent-0.index',
                'agent-1.data-00000-of-00001', 'agent-1.index', 'checkpoint'
            })

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
        self.assertTrue(all(episode_reward == 500.0 for episode_reward in runner.episode_rewards))
        runner.close()
        agent.close()
        self.finished_test()

        agent = Agent.load(
            directory='test/data', filename='ppo-checkpoint', format='numpy',
            environment=environment
        )
        runner = Runner(agent=agent, environment=environment, evaluation=True)
        runner.run(num_episodes=10, use_tqdm=False)
        self.assertTrue(all(episode_reward == 500.0 for episode_reward in runner.episode_rewards))
        runner.close()
        agent.close()
        self.finished_test()

        agent = Agent.load(
            directory='test/data', filename='ppo-checkpoint', format='hdf5',
            environment=environment
        )
        runner = Runner(agent=agent, environment=environment, evaluation=True)
        runner.run(num_episodes=10, use_tqdm=False)
        self.assertTrue(all(episode_reward == 500.0 for episode_reward in runner.episode_rewards))
        runner.close()
        agent.close()
        self.finished_test()

        agent = tf.saved_model.load(export_dir='test/data/ppo-checkpoint')

        # 10 episodes
        for _ in range(10):
            states = environment.reset()
            terminal = False
            episode_reward = 0.0
            while not terminal:
                states = np.expand_dims(states, axis=0)
                auxiliaries = dict(mask=np.ones(shape=(1, 2), dtype=bool))
                actions = agent.act(states, auxiliaries, True)
                actions = actions.numpy().item()
                states, terminal, reward = environment.execute(actions=actions)
                episode_reward += reward
            self.assertEqual(episode_reward, 500.0)

        environment.close()
        self.finished_test()
