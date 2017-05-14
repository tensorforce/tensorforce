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

"""
Simple runner for non-realtime single process execution, appropriate for
OpenAI gym.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange


class Runner(object):

    def __init__(self, agent, environment, preprocessor=None, repeat_actions=1):
        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions
        self.save_model_path = None
        self.save_model_episodes = 0
        self.episode_rewards = None

    def save_model(self, path, num_episodes):
        self.save_model_path = path
        self.save_model_episodes = num_episodes

    def run(self, episodes=-1, max_timesteps=-1, episode_finished=None, before_execution=None):
        self.episode_rewards = []  # save all episode rewards for statistics
        self.episode_lengths = []

        self.episode = 1
        while True:
            state = self.environment.reset()
            self.agent.reset()
            episode_reward = 0

            self.timestep = 1
            while True:
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                action = self.agent.act(state=processed_state)

                if before_execution:
                    action = before_execution(self, action)

                if self.repeat_actions > 1:
                    reward = 0
                    for repeat in xrange(self.repeat_actions):
                        state, step_reward, terminal = self.environment.execute(action=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, reward, terminal = self.environment.execute(action=action)

                episode_reward += reward
                self.agent.observe(state=processed_state, action=action, reward=reward, terminal=terminal)

                if terminal or self.timestep == max_timesteps:
                    break
                self.timestep += 1

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.timestep)

            if self.save_model_path and self.save_model_episodes > 0 and self.episode % self.save_model_episodes == 0:
                print("Saving agent after episode {}".format(self.episode))
                self.agent.save_model(self.save_model_path)

            if episode_finished and not episode_finished(self):
                return
            if self.episode == episodes:
                return
            self.episode += 1
