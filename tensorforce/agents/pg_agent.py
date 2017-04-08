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
Generic policy gradient agent. Manages batching and episodes internally, that is,
the only information needed is whether an episode ends.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce.config import create_config
from tensorforce.agents import RLAgent
from tensorforce.models import PGModel


class PGAgent(RLAgent):
    name = 'PGAgent'

    default_config = {}

    model = None

    def __init__(self, config, scope='pg_agent'):
        self.config = create_config(config, default=self.default_config)
        assert issubclass(self.__class__.model, PGModel)
        self.model = self.__class__.model(self.config, scope)
        self.continuous = self.config.continuous

        self.batch_size = self.config.batch_size
        self.current_batch = []
        self.batch_step = 0

        self.current_episode = self.model.zero_episode()
        self.episode_step = 0

        self.last_action = None
        self.last_action_means = None
        self.last_action_log_std = None

    def get_action(self, *args, **kwargs):
        """
        Executes one reinforcement learning step.

        :return: Which action to take
        """
        action, outputs = self.model.get_action(*args, **kwargs)

        # Cache last action in case action is used multiple times in environment
        self.last_action_means = outputs['policy_output']
        self.last_action = action

        if self.continuous:
            self.last_action_log_std = outputs['policy_log_std']
        else:
            action = np.argmax(action)

        return action

    def update(self, batch):
        """
        Explicitly calls update using the provided batch of experiences.

        :param batch:
        :return:
        """
        self.model.update(batch)

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

        self.current_episode['episode_length'] += 1
        self.current_episode['terminated'] = terminal
        self.current_episode['states'][self.episode_step] = state
        self.current_episode['actions'][self.episode_step] = self.last_action
        self.current_episode['action_means'][self.episode_step] = self.last_action_means
        self.current_episode['rewards'][self.episode_step] = reward
        if self.continuous:
            self.current_episode['action_log_stds'][self.episode_step] = self.last_action_log_std

        self.batch_step += 1
        self.episode_step += 1

        if terminal:
            # Transform into np arrays, append episode to batch, start new episode dict
            self.current_batch.append(self.current_episode)
            self.current_episode = self.model.zero_episode()
            self.episode_step = 0
            self.last_action = None
            self.last_action_means = None
            self.last_action_log_std = None

        if self.batch_step == self.batch_size:
            self.current_batch.append(self.current_episode)
            self.model.update(self.current_batch)
            self.current_episode = self.model.zero_episode()
            self.episode_step = 0
            self.current_batch = []
            self.batch_step = 0

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
