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

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

from test.unittest_base import UnittestBase
from test.unittest_environment import UnittestEnvironment


class TestDocumentation(UnittestBase, unittest.TestCase):

    def test_environment(self):
        self.start_tests(name='getting-started-environment')

        environment = Environment.create(
            environment='gym', level='CartPole', max_episode_timesteps=500
        )
        self.finished_test()

        environment = Environment.create(environment='gym', level='CartPole-v1')
        self.finished_test()

        environment = Environment.create(
            environment='test/data/environment.json', max_episode_timesteps=500
        )
        self.finished_test()

        environment = Environment.create(
            environment='test.data.custom_env.CustomEnvironment', max_episode_timesteps=10
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

        # Create agent and environment
        environment = Environment.create(
            environment='test/data/environment.json', max_episode_timesteps=10
        )
        agent = Agent.create(agent='test/data/agent.json', environment=environment)

        # Train for 200 episodes
        for _ in range(10):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

        # Evaluate for 100 episodes
        sum_rewards = 0.0
        for _ in range(5):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states, evaluation=True)
                states, terminal, reward = environment.execute(actions=actions)
                sum_rewards += reward

        sum_rewards / 100

        # Close agent and environment
        agent.close()
        environment.close()

        self.finished_test()

    def test_quickstart(self):
        self.start_tests(name='quickstart')

        # ====================

        # Create an OpenAI-Gym environment
        environment = Environment.create(environment='gym', level='CartPole-v1')

        # Create a PPO agent
        agent = Agent.create(
            agent='ppo', environment=environment,
            # Automatically configured network
            network='auto',
            # Optimization
            batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
            optimization_steps=5,
            # Reward estimation
            likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
            # Critic
            critic_network='auto',
            critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
            # Preprocessing
            preprocessing=None,
            # Exploration
            exploration=0.0, variable_noise=0.0,
            # Regularization
            l2_regularization=0.0, entropy_regularization=0.0,
            # TensorFlow etc
            name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
            summarizer=None, recorder=None
        )

        # Initialize the runner
        runner = Runner(agent=agent, environment=environment)

        # Start the runner
        runner.run(num_episodes=50, use_tqdm=False)
        runner.close()

        # ====================

        self.finished_test()

    def test_readme(self):
        self.start_tests(name='readme')

        environment = UnittestEnvironment(
            states=dict(type='float', shape=(10,)),
            actions=dict(type='int', shape=(), num_values=5),
            min_timesteps=5
        )

        def get_current_state():
            return environment.reset()

        def execute_decision(x):
            return environment.execute(actions=x)[2]

        # ==========

        from tensorforce.agents import Agent

        # Instantiate a Tensorforce agent
        agent = Agent.create(
            agent='tensorforce',
            states=dict(type='float', shape=(10,)),
            actions=dict(type='int', num_values=5),
            max_episode_timesteps=100,
            memory=10000,
            update=dict(unit='timesteps', batch_size=64),
            optimizer=dict(type='adam', learning_rate=3e-4),
            policy=dict(network='auto'),
            objective='policy_gradient',
            reward_estimation=dict(horizon=20)
        )

        # Retrieve the latest (observable) environment state
        state = get_current_state()  # (float array of shape [10])

        # Query the agent for its action decision
        action = agent.act(states=state)  # (scalar between 0 and 4)

        # Execute the decision and retrieve the current performance score
        reward = execute_decision(action)  # (any scalar float)

        # Pass feedback about performance (and termination) to the agent
        agent.observe(reward=reward, terminal=False)

        # ==========

        agent.close()
        environment.close()
        self.finished_test()

    def test_masking(self):
        self.start_tests(name='masking')

        agent, environment = self.prepare(
            states=dict(type='float', shape=(10,)),
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
