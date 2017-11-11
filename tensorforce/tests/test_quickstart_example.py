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

import sys
import unittest

import numpy as np
from six.moves import xrange

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


class TestQuickstartExample(unittest.TestCase):

    def test_example(self):
        sys.stdout.write('\nQuickstart:\n')
        sys.stdout.flush()

        passed = 0
        for _ in xrange(3):

            # Create an OpenAI-Gym environment
            environment = OpenAIGym('CartPole-v0')

            # Network specification for the model
            network_spec = [
                dict(type='dense', size=32),
                dict(type='dense', size=32)
            ]

            # Create the agent
            agent = PPOAgent(
                states_spec=environment.states,
                actions_spec=environment.actions,
                network_spec=network_spec,
                batch_size=4000,
                step_optimizer=dict(
                    type='adam',
                    learning_rate=1e-2
                ),
                optimization_steps=5,
                discount=0.99,
                normalize_rewards=False,
                entropy_regularization=0.01,
                likelihood_ratio_clipping = 0.2
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

            sys.stdout.write('episodes: {}\n'.format(runner.episode))
            sys.stdout.flush()

            # Test passed if episode_finished handle evaluated to False
            if runner.episode < 2000:
                passed += 1

        sys.stdout.write('==> passed: {}\n'.format(passed))
        sys.stdout.flush()
        self.assertTrue(passed >= 2)
