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

from tensorforce.util.experiment_util import repeat_action


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

    def run(self, episodes, max_timesteps, episode_finished=None):
        self.episode_rewards = []  # save all episode rewards for statistics

        for self.episode in xrange(episodes):
            state = self.environment.reset()
            episode_reward = 0

            for self.timestep in xrange(max_timesteps):
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                action = self.agent.get_action(processed_state, self.episode)
                result = repeat_action(self.environment, action, self.repeat_actions)

                episode_reward += result['reward']
                self.agent.add_observation(processed_state, action, result['reward'], result['terminal_state'])

                state = result['state']

                if result['terminal_state']:
                    break

            self.episode_rewards.append(episode_reward)

            if self.save_model_path and self.save_model_episodes > 0 and self.episode % self.save_model_episodes == 0:
                print("Saving agent after episode {}".format(self.episode))
                self.agent.save_model(self.save_model_path)

            if episode_finished and not episode_finished(self):
                return
