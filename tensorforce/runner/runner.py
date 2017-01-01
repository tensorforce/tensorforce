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


class Runner(object):

    def __init__(self, agent, environment, preprocessor=None, repeat_actions=1):
        self.agent = agent
        self.environment = environment
        self.preprocessor = preprocessor
        self.repeat_actions = repeat_actions

    def run(self, episodes, max_timesteps, episode_finished=None):
        self.total_states = 0      # count all states
        self.episode_rewards = []  # save all episode rewards for statistics

        for self.episode in xrange(episodes):
            state = self.environment.reset()
            episode_reward = 0
            repeat_action_count = 0

            for self.timestep in xrange(max_timesteps):
                if self.preprocessor:
                    processed_state = self.preprocessor.process(state)
                else:
                    processed_state = state

                # TODO: fix frame skip observations
                if repeat_action_count <= 0:
                    action = self.agent.get_action(processed_state, self.episode, self.total_states)
                    repeat_action_count = self.repeat_actions - 1
                    self.timestep += 1
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
            if episode_finished and not episode_finished(self):
                return
