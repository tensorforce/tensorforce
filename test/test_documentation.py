# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce import Agent, Environment, Runner

from test.unittest_base import UnittestBase


class TestDocumentation(UnittestBase, unittest.TestCase):

    def test_environment(self):
        self.start_tests(name='getting-started-environment')

        environment = Environment.create(
            environment='gym', level='CartPole', max_episode_timesteps=50
        )
        self.finished_test()

        environment = Environment.create(environment='gym', level='CartPole-v1')
        self.finished_test()

        environment = Environment.create(
            environment='test/data/environment.json', max_episode_timesteps=50
        )
        self.finished_test()

        environment = Environment.create(
            environment='test.data.custom_env.CustomEnvironment', max_episode_timesteps=10
        )
        self.finished_test()

        from test.data.custom_env import CustomEnvironment
        environment = Environment.create(
            environment=CustomEnvironment, max_episode_timesteps=10
        )
        self.finished_test()

    def test_agent(self):
        self.start_tests(name='getting-started-agent')

        environment = Environment.create(
            environment='gym', level='CartPole', max_episode_timesteps=50
        )
        self.finished_test()

        agent = Agent.create(
            agent='tensorforce', environment=environment, update=64,
            optimizer=dict(optimizer='adam', learning_rate=1e-3),
            objective='policy_gradient', reward_estimation=dict(horizon=20)
        )
        self.finished_test()

        agent = Agent.create(
            agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
        )
        self.finished_test()

        agent = Agent.create(agent='test/data/agent.json', environment=environment)
        self.finished_test()

    def test_execution(self):
        self.start_tests(name='getting-started-execution')

        runner = Runner(
            agent='test/data/agent.json', environment=dict(environment='gym', level='CartPole'),
            max_episode_timesteps=10
        )
        runner.run(num_episodes=10)
        runner.run(num_episodes=5, evaluation=True)
        runner.close()
        self.finished_test()

        runner = Runner(
            agent='test/data/agent.json', environment=dict(environment='gym', level='CartPole'),
            max_episode_timesteps=50, num_parallel=5, remote='multiprocessing'
        )
        runner.run(num_episodes=10)
        runner.close()
        self.finished_test()

        # Create agent and environment
        environment = Environment.create(
            environment='test/data/environment.json', max_episode_timesteps=10
        )
        agent = Agent.create(agent='test/data/agent.json', environment=environment)

        # Train for 100 episodes
        for _ in range(10):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

        # Train for 100 episodes
        for _ in range(10):
            episode_states = list()
            episode_internals = list()
            episode_actions = list()
            episode_terminal = list()
            episode_reward = list()

            states = environment.reset()
            internals = agent.initial_internals()
            terminal = False
            while not terminal:
                episode_states.append(states)
                episode_internals.append(internals)
                actions, internals = agent.act(states=states, internals=internals, independent=True)
                episode_actions.append(actions)
                states, terminal, reward = environment.execute(actions=actions)
                episode_terminal.append(terminal)
                episode_reward.append(reward)

            agent.experience(
                states=episode_states, internals=episode_internals, actions=episode_actions,
                terminal=episode_terminal, reward=episode_reward
            )
            agent.update()

        # Evaluate for 100 episodes
        sum_rewards = 0.0
        for _ in range(10):
            states = environment.reset()
            internals = agent.initial_internals()
            terminal = False
            while not terminal:
                actions, internals = agent.act(
                    states=states, internals=internals,
                    deterministic=True, independent=True
                )
                states, terminal, reward = environment.execute(actions=actions)
                sum_rewards += reward

        print('Mean episode reward:', sum_rewards / 100)

        # Close agent and environment
        agent.close()
        environment.close()

        self.finished_test()

    def test_readme(self):
        self.start_tests(name='readme')

        # ====================

        from tensorforce import Agent, Environment

        # Pre-defined or custom environment
        environment = Environment.create(
            environment='gym', level='CartPole', max_episode_timesteps=500
        )

        # Instantiate a Tensorforce agent
        agent = Agent.create(
            agent='tensorforce',
            environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
            memory=1000,
            update=dict(unit='timesteps', batch_size=64),
            optimizer=dict(type='adam', learning_rate=3e-4),
            policy=dict(network='auto'),
            objective='policy_gradient',
            reward_estimation=dict(horizon=20)
        )

        # Train for 300 episodes
        for _ in range(1):

            # Initialize episode
            states = environment.reset()
            terminal = False

            while not terminal:
                # Episode timestep
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

        agent.close()
        environment.close()

        # ====================

        self.finished_test()

    def test_modules(self):
        self.start_tests(name='modules')

        # distributions
        self.unittest(
            policy=dict(distributions=dict(
                float=dict(type='gaussian', global_stddev=True),
                bounded_action=dict(type='beta')
            ))
        )

        # layers
        import tensorflow as tf
        self.unittest(
            states=dict(type='float', shape=(2,), min_value=-1.0, max_value=2.0),
            policy=dict(network=[
                (lambda x: tf.clip_by_value(x, -1.0, 1.0)),
                dict(type='dense', size=8, activation='tanh')
            ])
        )

        # memories
        self.unittest(
            memory=100
        )

        # networks
        self.unittest(
            states=dict(type='float', shape=(2,), min_value=1.0, max_value=2.0),
            policy=dict(network=[
                dict(type='dense', size=8, activation='tanh'),
                dict(type='dense', size=8, activation='tanh')
            ])
        )
        self.unittest(
            states=dict(
                observation=dict(type='float', shape=(4, 4, 3), min_value=-1.0, max_value=2.0),
                attributes=dict(type='int', shape=(4, 2), num_values=5)
            ),
            policy=[
                [
                    dict(type='retrieve', tensors=['observation']),
                    dict(type='conv2d', size=8),
                    dict(type='flatten'),
                    dict(type='register', tensor='obs-embedding')
                ],
                [
                    dict(type='retrieve', tensors=['attributes']),
                    dict(type='embedding', size=8),
                    dict(type='flatten'),
                    dict(type='register', tensor='attr-embedding')
                ],
                [
                    dict(
                        type='retrieve', tensors=['obs-embedding', 'attr-embedding'],
                        aggregation='concat'
                    ),
                    dict(type='dense', size=16)
                ]
            ]
        )

        # optimizers
        self.unittest(
            optimizer=dict(
                optimizer='adam', learning_rate=1e-3, clipping_threshold=1e-2,
                multi_step=3, subsampling_fraction=8, linesearch_iterations=3,
                doublecheck_update=True
            )
        )

        # parameters
        self.unittest(
            exploration=0.1
        )
        self.unittest(
            optimizer=dict(optimizer='adam', learning_rate=dict(
                type='exponential', unit='timesteps', num_steps=2,
                initial_value=0.01, decay_rate=0.5
            ))
        )
        self.unittest(
            reward_estimation=dict(horizon=dict(
                type='linear', unit='episodes', num_steps=2,
                initial_value=2, final_value=6
            ))
        )

        # preprocessing
        self.unittest(
            states=dict(type='float', shape=(8, 8, 3), min_value=-1.0, max_value=2.0),
            state_preprocessing=[
                dict(type='image', height=4, width=4, grayscale=True),
                dict(type='exponential_normalization', decay=0.999)
            ],
            reward_preprocessing=dict(type='clipping', lower=-1.0, upper=1.0)
        )

        # policy
        self.unittest(
            states=dict(type='float', shape=(2,), min_value=-1.0, max_value=2.0),
            policy=[
                dict(type='dense', size=8, activation='tanh'),
                dict(type='dense', size=8, activation='tanh')
            ]
        )
        self.unittest(
            states=dict(type='float', shape=(2,), min_value=-1.0, max_value=2.0),
            policy=dict(network='auto')
        )
        self.unittest(
            states=dict(type='float', shape=(2,), min_value=-1.0, max_value=2.0),
            policy=dict(
                type='parametrized_distributions',
                network=[
                    dict(type='dense', size=8, activation='tanh'),
                    dict(type='dense', size=8, activation='tanh')
                ],
                distributions=dict(
                    float=dict(type='gaussian', global_stddev=True),
                    bounded_action=dict(type='beta')
                ),
                temperature=dict(
                    type='decaying', decay='exponential', unit='episodes',
                    num_steps=2, initial_value=0.01, decay_rate=0.5
                )
            )
        )

    def test_masking(self):
        self.start_tests(name='masking')

        agent, environment = self.prepare(
            states=dict(type='float', shape=(10,), min_value=-1.0, max_value=2.0),
            actions=dict(type='int', shape=(), num_values=3)
        )
        states = environment.reset()
        assert 'state' in states and 'action_mask' in states
        states['action_mask'] = [True, False, True]

        action = agent.act(states=states)
        assert action != 1

        agent.close()
        environment.close()
        self.finished_test()
