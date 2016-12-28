# Copyright 2016 reinforce.io. All Rights Reserved.
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

"""
Runner base class
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np


class Runner(object):

    def __init__(self, agent, environment, preprocessor=None, repeat_actions=1):
        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions

    def run(self, episodes, max_timesteps, report=True, report_episodes=None):
        total_states = 0      # count all states
        episode_rewards = []  # save all episode rewards for statistics
        if not report_episodes:
            report_episodes = episodes // 100

        if report:
            print("Starting {agent} for Environment '{env}'".format(agent=self.agent, env=self.environment))
        for episode_num in xrange(episodes):
            state = self.environment.reset()
            episode_reward = 0
            repeat_action_count = 0

            for timestep_num in xrange(max_timesteps):
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                if repeat_action_count <= 0:
                    action = self.agent.get_action(processed_state, episode_num, total_states)
                    repeat_action_count = self.repeat_actions - 1
                else:
                    repeat_action_count -= 1

                result = self.environment.execute_action(action)

                episode_reward += result['reward']
                self.agent.add_observation(processed_state, action, result['reward'], result['terminal_state'])

                state = result['state']
                total_states += 1

                if result['terminal_state']:
                    break

            episode_rewards.append(episode_reward)

            if report and episode_num % report_episodes == 0:
                print("Finished episode {ep} after {ts} timesteps".format(ep=episode_num + 1, ts=timestep_num + 1))
                print("Episode reward: {}".format(episode_reward))
                print("Average of last 500 rewards: {}".format(np.mean(episode_rewards[-500:])))
                print("Average of last 100 rewards: {}".format(np.mean(episode_rewards[-100:])))
        print("Learning finished. Total episodes: {ep}".format(ep=episode_num + 1))