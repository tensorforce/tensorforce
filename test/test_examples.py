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

import os
from tempfile import TemporaryDirectory
import unittest

from tensorforce import Agent, Environment, Runner

from test.unittest_base import UnittestBase


class TestExamples(UnittestBase, unittest.TestCase):

    def test_quickstart(self):
        self.start_tests(name='quickstart')

        with TemporaryDirectory() as saver_directory, TemporaryDirectory() as summarizer_directory:

            # ====================

            # OpenAI-Gym environment specification
            environment = dict(environment='gym', level='CartPole-v1')
            # or: environment = Environment.create(
            #         environment='gym', level='CartPole-v1', max_episode_timesteps=500)

            # PPO agent specification
            agent = dict(
                agent='ppo',
                # Automatically configured network
                network='auto',
                # PPO optimization parameters
                batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
                subsampling_fraction=0.33,
                # Reward estimation
                likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
                # Baseline network and optimizer
                baseline_network=dict(type='auto', size=32, depth=1),
                baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
                # Preprocessing
                preprocessing='linear_normalization',
                # Exploration
                exploration=0.0, variable_noise=0.0,
                # Regularization
                l2_regularization=0.0, entropy_regularization=0.0,
                # Default additional config values
                config=None,
                # Save model every 10 updates and keep the 5 most recent checkpoints
                saver=dict(directory=saver_directory, frequency=10, max_checkpoints=5),
                # Log all available Tensorboard summaries
                summarizer=dict(directory=summarizer_directory, labels='all'),
                # Do not record agent-environment interaction trace
                recorder=None
            )
            # or: Agent.create(agent='ppo', environment=environment, ...)
            # with additional argument "environment" and, if applicable, "parallel_interactions"

            # Initialize the runner
            runner = Runner(agent=agent, environment=environment, max_episode_timesteps=500)

            # Start the runner
            runner.run(num_episodes=20)
            runner.close()

            # plus agent.close() and environment.close() if created separately

            # ====================

            files = set(os.listdir(path=saver_directory))
            self.assertTrue(files == {
                'agent.json', 'agent-0.data-00000-of-00001', 'agent-0.index',
                'agent-10.data-00000-of-00001', 'agent-10.index', 'checkpoint'
            })

            directories = os.listdir(path=summarizer_directory)
            self.assertEqual(len(directories), 1)
            files = os.listdir(path=os.path.join(summarizer_directory, directories[0]))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith('events.out.tfevents.'))

        self.finished_test()

    def test_record_and_pretrain(self):
        self.start_tests(name='record-and-pretrain')

        with TemporaryDirectory() as directory:

            # ====================

            # Start recording traces after the first 100 episodes -- by then, the agent
            # has solved the environment
            runner = Runner(
                agent=dict(
                    agent='benchmarks/configs/ppo.json',
                    recorder=dict(directory=directory, start=8)
                ), environment='benchmarks/configs/cartpole.json'
            )
            runner.run(num_episodes=10)
            runner.close()

            # Pretrain a new agent on the recorded traces: for 30 iterations, feed the
            # experience of one episode to the agent and subsequently perform one update
            environment = Environment.create(environment='benchmarks/configs/cartpole.json')
            agent = Agent.create(agent='benchmarks/configs/ppo.json', environment=environment)
            agent.pretrain(
                directory='test/data/ppo-traces', num_iterations=30, num_traces=1, num_updates=1
            )

            # Evaluate the pretrained agent
            runner = Runner(agent=agent, environment=environment)
            runner.run(num_episodes=10, evaluation=True)
            self.assertTrue(
                all(episode_reward == 500.0 for episode_reward in runner.episode_rewards)
            )
            runner.close()

            # Close agent and environment
            agent.close()
            environment.close()

            # ====================

            files = sorted(os.listdir(path=directory))
            self.assertEqual(len(files), 2)
            self.assertTrue(all(
                file.startswith('trace-') and file.endswith('0000000{}.npz'.format(n))
                for n, file in enumerate(files, start=8)
            ))

        self.finished_test()

    def test_temperature_controller(self):
        self.start_tests(name='temperature-controller')

        # ====================

        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        ## Compute the response for a given action and current temperature
        def respond(action, current_temp, tau):
            return action + (current_temp - action) * math.exp(-1.0/tau)

        ## Actions of a series of on, then off
        sAction = pd.Series(np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]))
        sResponse = np.zeros(sAction.size)

        ## Update the response with the response to the action
        for i in range(sAction.size):
            ## Get last response
            if i == 0:
                last_response = 0
            else:
                last_response = sResponse[i - 1]
            sResponse[i] = respond(sAction[i], last_response, 3.0)

        ## Assemble and plot
        df = pd.DataFrame(list(zip(sAction, sResponse)), columns=['action', 'response'])
        df.plot()

        # ====================

        def reward(temp):
                delta = abs(temp - 0.5)
                if delta < 0.1:
                    return 0.0
                else:
                    return -delta + 0.1

        temps = [x * 0.01 for x in range(100)]
        rewards = [reward(x) for x in temps]

        fig=plt.figure(figsize=(12, 4))

        plt.scatter(temps, rewards)
        plt.xlabel('Temperature')
        plt.ylabel('Reward')
        plt.title('Reward vs. Temperature')

        # ====================

        ###-----------------------------------------------------------------------------
        ## Imports
        from tensorforce.environments import Environment
        from tensorforce.agents import Agent

        ###-----------------------------------------------------------------------------
        ### Environment definition
        class ThermostatEnvironment(Environment):
            """This class defines a simple thermostat environment.  It is a room with
            a heater, and when the heater is on, the room temperature will approach
            the max heater temperature (usually 1.0), and when off, the room will
            decay to a temperature of 0.0.  The exponential constant that determines
            how fast it approaches these temperatures over timesteps is tau.
            """
            def __init__(self):
                ## Some initializations.  Will eventually parameterize this in the constructor.
                self.tau = 3.0
                self.current_temp = np.random.random(size=(1,))

                super().__init__()

            def states(self):
                return dict(type='float', shape=(1,))

            def actions(self):
                """Action 0 means no heater, temperature approaches 0.0.  Action 1 means
                the heater is on and the room temperature approaches 1.0.
                """
                return dict(type='int', num_values=2)

            # Optional, should only be defined if environment has a natural maximum
            # episode length
            def max_episode_timesteps(self):
                return super().max_episode_timesteps()

            # Optional
            def close(self):
                super().close()

            def reset(self):
                """Reset state.
                """
                # state = np.random.random(size=(1,))
                self.timestep = 0
                self.current_temp = np.random.random(size=(1,))
                return self.current_temp

            def response(self, action):
                """Respond to an action.  When the action is 1, the temperature
                exponentially decays approaches 1.0.  When the action is 0,
                the current temperature decays towards 0.0.
                """
                return action + (self.current_temp - action) * math.exp(-1.0 / self.tau)

            def reward_compute(self):
                """ The reward here is 0 if the current temp is between 0.4 and 0.6,
                else it is distance the temp is away from the 0.4 or 0.6 boundary.
                
                Return the value within the numpy array, not the numpy array.
                """
                delta = abs(self.current_temp - 0.5)
                if delta < 0.1:
                    return 0.0
                else:
                    return -delta[0] + 0.1

            def execute(self, actions):
                ## Check the action is either 0 or 1 -- heater on or off.
                assert actions == 0 or actions == 1

                ## Increment timestamp
                self.timestep += 1
                
                ## Update the current_temp
                self.current_temp = self.response(actions)
                
                ## Compute the reward
                reward = self.reward_compute()

                ## The only way to go terminal is to exceed max_episode_timestamp.
                ## terminal == False means episode is not done
                ## terminal == True means it is done.
                terminal = False

                return self.current_temp, terminal, reward

        ###-----------------------------------------------------------------------------
        ### Create the environment
        ###   - Tell it the environment class
        ###   - Set the max timestamps that can happen per episode
        environment = environment = Environment.create(
            environment=ThermostatEnvironment,
            max_episode_timesteps=100)

        # ====================

        agent = Agent.create(
            agent='tensorforce', environment=environment, update=64,
            objective='policy_gradient', reward_estimation=dict(horizon=1)
        )

        # ====================

        ### Initialize
        environment.reset()

        ## Creation of the environment via Environment.create() creates
        ## a wrapper class around the original Environment defined here.
        ## That wrapper mainly keeps track of the number of timesteps.
        ## In order to alter the attributes of your instance of the original
        ## class, like to set the initial temp to a custom value, like here,
        ## you need to access the `environment` member of this wrapped class.
        ## That is why you see the way to set the current_temp like below.
        environment.current_temp = np.array([0.5])
        states = environment.current_temp

        internals = agent.initial_internals()
        terminal = False

        ### Run an episode
        temp = [environment.current_temp[0]]
        while not terminal:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            temp += [states[0]]

        ### Plot the run
        plt.figure(figsize=(12, 4))
        ax=plt.subplot()
        ax.set_ylim([0.0, 1.0])
        plt.plot(range(len(temp)), temp)
        plt.hlines(y=0.4, xmin=0, xmax=99, color='r')
        plt.hlines(y=0.6, xmin=0, xmax=99, color='r')
        plt.xlabel('Timestep')
        plt.ylabel('Temperature')
        plt.title('Temperature vs. Timestep')
        plt.show()

        # Train for 200 episodes
        for _ in range(50):
            states = environment.reset()
            terminal = False
            while not terminal:
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

        # ====================

        ### Initialize
        environment.reset()

        ## Creation of the environment via Environment.create() creates
        ## a wrapper class around the original Environment defined here.
        ## That wrapper mainly keeps track of the number of timesteps.
        ## In order to alter the attributes of your instance of the original
        ## class, like to set the initial temp to a custom value, like here,
        ## you need to access the `environment` member of this wrapped class.
        ## That is why you see the way to set the current_temp like below.
        environment.current_temp = np.array([1.0])
        states = environment.current_temp

        internals = agent.initial_internals()
        terminal = False

        ### Run an episode
        temp = [environment.current_temp[0]]
        while not terminal:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            temp += [states[0]]

        ### Plot the run
        plt.figure(figsize=(12, 4))
        ax=plt.subplot()
        ax.set_ylim([0.0, 1.0])
        plt.plot(range(len(temp)), temp)
        plt.hlines(y=0.4, xmin=0, xmax=99, color='r')
        plt.hlines(y=0.6, xmin=0, xmax=99, color='r')
        plt.xlabel('Timestep')
        plt.ylabel('Temperature')
        plt.title('Temperature vs. Timestep')
        plt.show()

        # ====================

        self.finished_test()
