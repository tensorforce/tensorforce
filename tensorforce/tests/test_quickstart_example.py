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

import unittest

import numpy as np
from six.moves import xrange

from tensorforce import Configuration
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


class TestQuickstartExample(unittest.TestCase):

    def test_example(self):
        passed = 0

        for _ in xrange(3):
            # Create an OpenAIgym environment
            env = OpenAIGym('CartPole-v0')
            config = Configuration(
                batch_size=4096,
                # Agent
                preprocessing=None,
                exploration=None,
                reward_preprocessing=None,
                # BatchAgent
                keep_last_timestep=True,  # not documented!
                # PPOAgent
                step_optimizer=dict(
                    type='adam',
                    learning_rate=1e-3
                ),
                optimization_steps=10,
                # Model
                scope='ppo',
                discount=0.99,
                # DistributionModel
                distributions=None,  # not documented!!!
                entropy_regularization=0.01,
                # PGModel
                baseline_mode=None,
                baseline=None,
                baseline_optimizer=None,
                gae_lambda=None,
                normalize_rewards=False,
                # PGLRModel
                likelihood_ratio_clipping=0.2,
                # Logging
                log_level='info',
                # TensorFlow Summaries
                summary_logdir=None,
                summary_labels=['total-loss'],
                summary_frequency=1,
                # Distributed
                distributed=False,
                device=None
            )

            network_spec = [
                dict(type='dense', size=32, activation='tanh'),
                dict(type='dense', size=32, activation='tanh')
            ]
            # Create a Trust Region Policy Optimization agent
            agent = PPOAgent(
                states_spec=env.states,
                actions_spec=env.actions,
                network_spec=network_spec,
                config=config
            )
            runner = Runner(agent=agent, environment=env)

            def episode_finished(r):
                # Test if mean reward over 50 should ensure that learning took off
                avg_reward = np.mean(r.episode_rewards[-50:])
                return r.episode < 100 or avg_reward < 50.0

            runner.run(episodes=2000, max_timesteps=200, episode_finished=episode_finished)

            if runner.episode < 2000:
                passed += 1

        print('Quick start example passed = {}'.format(passed))
        self.assertTrue(passed >= 2)
