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
Generic policy gradient agent. Manages batching and episodes internally, that is,
the only information needed is whether an episode ends.
"""
from collections import defaultdict
from copy import deepcopy
import numpy as np
from tensorforce.config import create_config
from tensorforce.rl_agents import RLAgent


class PGAgent(RLAgent):
    name = 'PGAgent'

    default_config = {
        'batch_size': 5000,
        'deterministic_mode': False,
    }

    model_ref = None

    def __init__(self, config):
        self.config = create_config(config, default=self.default_config)
        self.updater = None
        self.current_batch = []
        self.current_episode = defaultdict(list)
        self.batch_steps = 0
        self.batch_size = config.batch_size
        self.last_action = None
        self.last_action_means = None
        self.last_action_log_std = None
        self.continuous = self.config.continuous

        if self.model_ref:
            self.updater = self.model_ref(self.config)

    def get_action(self, *args, **kwargs):
        """
        Executes one reinforcement learning step.

        :param state: Observed state tensor
        :param episode: Optional, current episode
        :return: Which action to take
        """
        action, outputs = self.updater.get_action(*args, **kwargs)
        #print(outputs)
        # Cache last action in case action is used multiple times in environment
        self.last_action_means = outputs['policy_output']
        self.last_action = action


        #print('action one hot =' + str(action))

        if self.continuous:
            self.last_action_log_std = outputs['policy_log_std']
        else:
            action = np.argmax(action)


        #print('action selected' + str(action))
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
        self.current_episode['actions'].append(self.last_action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['action_means'].append(self.last_action_means)

        if self.continuous:
            self.current_episode['action_log_stds'].append(self.last_action_log_std)

        if terminal:
            # Batch could also end before episode is terminated
            self.current_episode['terminated'] = True

            # Transform into np arrays, append episode to batch, start new episode dict
            path = self.get_path()
            self.current_batch.append(path)
            self.current_episode = defaultdict(list)

        if self.batch_steps == self.batch_size:
            if not terminal:
                self.current_episode['terminated'] = False
                path = self.get_path()
                self.current_batch.append(path)

            #print('last means=' + str(self.last_action_means))
            #if self.continuous:
                #print('last stds=' + str(self.last_action_log_std))
            #print('last actions=' + str(self.last_action))

            print('Computing PG update, episodes =' + str(len(self.current_batch)))
            self.updater.update(self.current_batch)
            self.current_episode = defaultdict(list)
            self.current_batch = []
            self.last_action = None
            self.last_action_means = None
            self.last_action_log_std = None
            self.batch_steps = 0

    def get_path(self):
        """
        Finalises an episode and turns it into a dict pointing to numpy arrays.
        :return:
        """

        path = {'states': np.concatenate(np.expand_dims(self.current_episode['states'], 0)),
                'actions': np.array(self.current_episode['actions']),
                'terminated': self.current_episode['terminated'],
                'action_means': np.array(self.current_episode['action_means']),
                'rewards': np.array(self.current_episode['rewards'])}

        if self.continuous:
            path['action_log_stds'] = np.concatenate(self.current_episode['action_log_stds'])

        return path

    def save_model(self, path):
        self.updater.save_model(path)

    def load_model(self, path):
        self.updater.load_model(path)
