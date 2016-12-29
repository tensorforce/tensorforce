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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange

from tensorforce.config import create_config

class Runner(object):

    default_config = {
        'episodes': 10000,
        'max_timesteps': 2000,
        'repeat_actions': 4
    }

    def __init__(self, config, agent, environment, preprocessor=None):
        self.config = create_config(config, default=self.default_config)
        self.episodes = self.config.episodes
        self.max_timesteps = self.config.max_timesteps
        self.repeat_actions = self.config.repeat_actions

        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor

    def run(self, episode_finished=None):
        self.total_states = 0
        self.episode_rewards = []

        for self.episode in xrange(self.episodes):
            state = self.environment.reset()
            episode_reward = 0
            repeat_action_count = 0

            for j in xrange(self.max_timesteps):
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state
                if repeat_action_count <= 0:
                    action = self.agent.get_action(processed_state, episode=self.episode, total_states=self.total_states)
                    repeat_action_count = self.repeat_actions - 1
                else:
                    repeat_action_count -= 1

                result = self.environment.execute_action(action)
                episode_reward += result['reward']
                self.agent.add_observation(processed_state, action, result['reward'], result['terminal_state'])

                state = result['state']
                self.total_states += 1
                if result['terminal_state']:
                    break

            self.episode_rewards.append(episode_reward)
            print('episode finished', self.episode_rewards)
            if episode_finished and not episode_finished(self):
                print('end')
                return
