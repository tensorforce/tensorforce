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
Generic policy gradient agent.
"""
from collections import defaultdict
from copy import deepcopy

from tensorforce.config import create_config
from tensorforce.rl_agents.rl_agent import RLAgent


class PGAgent(RLAgent):

    default_config = {
        'batch_size': 10000,
        'deterministic_mode': False,
        'gamma': 0.99
    }

    value_function_ref = None


    def __init__(self, config):

        self.config = create_config(config, default=self.default_config)
        self.updater = None
        self.current_batch = []
        self.current_episode = defaultdict(list)
        self.batch_steps = 0
        self.batch_size = config.batch_size

        if self.value_function_ref:
            self.updater = self.value_function_ref(self.config)

    def get_action(self, state, episode=1):
        """
        Executes one reinforcement learning step.

        :param state: Observed state tensor
        :param episode: Optional, current episode
        :return: Which action to take
        """

        action, outputs = self.updater.get_action(state, episode)
        # TODO this assumes we always call get action/add observation together, need safeguards
        self.current_episode['action_means'].append(outputs['action_means'])
        self.current_episode['action_log_stds'].append(outputs['action_log_stds'])

        return action


    def add_observation(self, state, action, reward, terminal):
        """
        Adds an observation and performs a pg update if the necessary conditions
        are satisfied, i.e. if one batch of experience has been collected as defined
        by the batch size.

        In particular, note that episode control happens outside of the agent since
        the agent should be agnostic to how the training data is created.

        :param state:
        :param action:
        :param reward:
        :param terminal:
        :return:
        """

        self.batch_steps += 1
        self.current_episode['states'].append(state)
        self.current_episode['actions'].append(action)
        self.current_episode['reward'].append(reward)

        if terminal:
            # Batch could also end before episode is terminated
            self.current_episode['terminated'] = True

            # Append episode to batch, start new episode dict
            self.current_batch.append(deepcopy(self.current_episode))
            self.current_episode = defaultdict(list)

        if self.batch_steps == self.batch_size:
            self.updater.update(deepcopy(self.current_batch))
            self.current_episode = defaultdict(list)
            self.current_batch = []
            self.batch_steps = 0

    def save_model(self, path):
        self.updater.save_model(path)

    def load_model(self, path):
        self.updater.load_model(path)
