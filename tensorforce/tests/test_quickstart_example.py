# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from six.moves import xrange
import sys
import unittest

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


logging.getLogger('tensorflow').disabled = True


class TestQuickstartExample(unittest.TestCase):

    def test_example(self):
        sys.stdout.write('\nQuickstart:\n')
        sys.stdout.flush()

        passed = 0
        for _ in xrange(3):

            # Create an OpenAI-Gym environment
            environment = OpenAIGym('CartPole-v0')

            # Network specification for the model
            network = [
                dict(type='dense', size=32),
                dict(type='dense', size=32)
            ]

            # Create the agent
            agent = PPOAgent(
                states=environment.states,
                actions=environment.actions,
                network=network,
                # Model
                states_preprocessing=None,
                actions_exploration=None,
                reward_preprocessing=None,
                # MemoryModel
                update_mode=dict(
                    unit='episodes',
                    # 10 episodes per update
                    batch_size=10,
                    # Every 10 episodes
                    frequency=10
                ),
                memory=dict(
                    type='latest',
                    include_next_states=False,
                    capacity=5000
                ),
                discount=0.99,
                # DistributionModel
                distributions=None,
                entropy_regularization=0.01,
                # PGModel
                baseline_mode='states',
                baseline=dict(
                    type='mlp',
                    sizes=[32, 32]
                ),
                baseline_optimizer=dict(
                    type='multi_step',
                    optimizer=dict(
                        type='adam',
                        learning_rate=1e-3
                    ),
                    num_steps=5
                ),
                gae_lambda=None,
                # PGLRModel
                likelihood_ratio_clipping=0.2,
                # PPOAgent
                step_optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                subsampling_fraction=0.1,
                optimization_steps=50
            )

            # Initialize the runner
            runner = Runner(agent=agent, environment=environment)

            # Function handle called after each finished episode
            def episode_finished(r):
                # Test if mean reward over 50 should ensure that learning took off
                mean_reward = np.mean(r.episode_rewards[-50:])
                return r.episode < 100 or mean_reward < 50.0

            # Start the runner
            runner.run(episodes=2000, max_episode_timesteps=200, episode_finished=episode_finished)
            runner.close()

            sys.stdout.write('episodes: {}\n'.format(runner.episode))
            sys.stdout.flush()

            # Test passed if episode_finished handle evaluated to False
            if runner.episode < 2000:
                passed += 1

        sys.stdout.write('==> passed: {}\n'.format(passed))
        sys.stdout.flush()
        self.assertTrue(passed >= 2)
