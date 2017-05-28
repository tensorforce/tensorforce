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
Generic agent for distributed real time training.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from copy import deepcopy

from tensorforce.models.distributed_pg_model import DistributedPGModel


class DistributedAgent(object):
    name = 'DistributedAgent'
    default_config = {}

    model = None

    def __init__(self, config, network_config, scope, task_index, cluster_spec):
        self.config = create_config(config, default=self.default_config)

        self.continuous = self.config.continuous

        self.model = DistributedPGModel(config, network_config, scope, task_index, cluster_spec)

        self.current_episode = self.model.zero_episode()
        self.current_episode['terminated'] = False
        self.current_experience = Experience(self.continuous, self.model.zero_episode())

    def get_global_step(self):
        return self.model.get_global_step()

    def update(self):
        """
        Updates the model using the given batch of experiences.

        """

        # Just one episode, but model logic expects list of episodes in case batch
        # spans multiple episodes
        batch = [self.current_episode]
        self.model.update(deepcopy(batch))

        # Reset current episode
        self.current_episode = self.model.zero_episode()
        self.current_episode['terminated'] = False

    def extend(self, experience):
        self.current_episode['episode_length'] += experience.data['episode_length']
        self.current_episode['states'] += experience.data['states']
        self.current_episode['actions'] += experience.data['actions']
        self.current_episode['rewards'] += experience.data['rewards']
        self.current_episode['action_means'] += experience.data['action_means']
        self.current_episode['terminated'] = experience.data['terminated']

        if self.continuous:
            self.current_episode['action_log_stds'] += experience.current_episode['action_log_stds']

    def sync(self):
        self.model.sync_global_to_local()

    def get_action(self, *args, **kwargs):
        """
        Executes one reinforcement learning step.

        :param state: Observed state tensor
        :param episode: Optional, current episode
        :return: Which action to take
        """
        experience = kwargs.pop('experience', None)
        action, outputs = self.model.get_action(*args, **kwargs)

        # Cache last action in case action is used multiple times in environment
        experience.last_action_means = outputs['policy_output']
        experience.last_action = action

        if self.continuous:
            experience.last_action_log_std = outputs['policy_log_std']
        else:
            action = np.argmax(action)

        return action

    def load_model(self, path):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def set_session(self, session):
        self.model.set_session(session)

    def __str__(self):
        return self.name


class Experience(object):
    """
    Helper object for queue management.
    """

    def __init__(self, continuous, data):
        self.continuous = continuous
        self.data = data
        self.episode_step = 0
        self.last_action = None
        self.last_action_means = None
        self.last_action_log_std = None
        self.data['terminated'] = False

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

        self.data['episode_length'] += 1
        self.data['terminated'] = terminal
        self.data['states'][self.episode_step] = state
        self.data['actions'][self.episode_step] = self.last_action
        self.data['action_means'][self.episode_step] = self.last_action_means
        self.data['rewards'][self.episode_step] = reward

        if self.continuous:
            self.data['action_log_stds'][self.episode_step] = self.last_action_log_std

        self.episode_step += 1