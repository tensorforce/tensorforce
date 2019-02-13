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

import logging
import numpy as np
import sys
import unittest

from tensorforce.agents import PPOAgent
from tensorforce.environments import OpenAIGym
from tensorforce.execution import Runner


logging.getLogger('tensorflow').disabled = True


class TestQuickstart(unittest.TestCase):

    def test_quickstart(self):
        sys.stdout.write('\nQuickstart:\n')
        sys.stdout.flush()

        # Create an OpenAI-Gym environment
        environment = OpenAIGym('CartPole-v1')

        # Create the agent
        agent = PPOAgent(
            states=environment.states(), actions=environment.actions(),
            # Automatically configured network
            network='auto',
            # Memory sampling most recent experiences, with a capacity of 2500 timesteps
            # (6100 > [30 batch episodes] * [200 max timesteps per episode])
            memory=6100,
            # Update every 10 episodes, with a batch of 30 episodes
            update_mode=dict(unit='episodes', batch_size=30, frequency=10),
            # PPO optimizer
            step_optimizer=dict(type='adam', learning_rate=1e-3),
            # PPO multi-step optimization: 10 updates, each based on a third of the batch
            subsampling_fraction=0.33, optimization_steps=10,
            # MLP baseline
            baseline_mode='states', baseline=dict(type='network', network='auto'),
            # Baseline optimizer
            baseline_optimizer=dict(
                type='multi_step', optimizer=dict(type='adam', learning_rate=1e-4), num_steps=5
            ),
            # Other parameters
            discount=0.99, entropy_regularization=1e-2, gae_lambda=None, likelihood_ratio_clipping=0.2
        )

        # Initialize the runner
        runner = Runner(agent=agent, environment=environment)

        # Function handle called after each finished episode
        def callback(r):
            return float(np.mean(r.episode_rewards[-100:])) <= 180.0

        # Start the runner
        runner.run(num_episodes=1000, max_episode_timesteps=200, callback=callback)
        runner.close()

        if float(np.mean(runner.episode_rewards[-100:])) <= 180.0:
            sys.stdout.write('Test failed, exceeding {} episodes\n'.format(runner.episode))
            sys.stdout.flush()
            self.assertTrue(expr=False)
        else:
            sys.stdout.write('Test passed after {} episodes\n'.format(runner.episode))
            sys.stdout.flush()
            self.assertTrue(expr=True)
