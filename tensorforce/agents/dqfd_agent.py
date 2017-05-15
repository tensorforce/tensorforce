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
Deep Q-learning from demonstration. This agent pre-trains from demonstration data.
 
Original paper: 'Learning from Demonstrations for Real World Reinforcement Learning'

https://arxiv.org/abs/1704.03732
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from tensorforce.config import create_config
from tensorforce.core import Agent


class DQFDAgent(Agent):

    name = 'DQFDAgent'
    model = DQFDModel

    default_config = DQFDAgentConfig

    def __init__(self, config, scope='dqfd_agent', network_builder=None):
        """
        
        :param config: 
        :param scope: 
        """
        self.config = create_config(config, default=self.default_config)

        # This is the online memory
        self.replay_memory = ReplayMemory(**self.config)

        # This is the demonstration memory that we will fill with observations before starting
        # the main training loop
        # TODO we might want different sizes for these memories -> add config param
        self.demo_memory = ReplayMemory(**self.config)

        self.step_count = 0

        # Called p in paper, controls ratio of expert vs online training samples
        self.expert_sampling_ratio = self.config.expert_sampling_ratio


        self.update_repeat = self.config.update_repeat
        self.batch_size = self.config.batch_size

        # p = n_demo / (n_demo + n_replay) => n_demo  = p * n_replay / (1 - p)
        self.demo_batch_size = int(self.expert_sampling_ratio * self.batch_size / \
                               (1.0 - self.expert_sampling_ratio))
        self.update_steps = int(round(1 / self.config.update_rate))
        self.use_target_network = self.config.use_target_network

        if self.use_target_network:
            self.target_update_steps = int(round(1 / self.config.target_network_update_rate))

        self.min_replay_size = self.config.min_replay_size

        if self.__class__.model:
            self.model = self.__class__.model(self.config, scope, network_builder=network_builder)

    def add_demo_observation(self, state, action, reward, terminal):
        """
        Adds observations to demo memory. 

        """
        self.demo_memory.add_experience(state, action, reward, terminal)

    def pre_train(self, steps=1):
        """
        
        :param steps: Number of pre-train updates to perform.
        
        """
        for _ in xrange(steps):
            # Sample from demo memory
            batch = self.demo_memory.sample_batch(self.batch_size)

            # Update using both double Q-learning and supervised double_q_loss
            self.model.pre_train_update(batch)

    def add_observation(self, state, action, reward, terminal):
        """
        Adds observations, updates via sampling from memories according to update rate.
        In the DQFD case, we sample from the online replay memory and the demo memory with
        the fractions controlled by a hyperparameter p called 'expert sampling ratio.
        
        :param state: 
        :param action: 
        :param reward: 
        :param terminal: 
        :return: 
        """
        self.replay_memory.add_experience(state, action, reward, terminal)

        self.step_count += 1

        if self.step_count >= self.min_replay_size and self.step_count % self.update_steps == 0:
            for _ in xrange(self.update_repeat):
                # Sample batches according to expert sampling ratio
                # In the paper, p is given as p = n_demo / (n_replay + n_demo)
                demo_batch = self.demo_memory.sample_batch(self.demo_batch_size)
                online_batch = self.replay_memory.sample_batch(self.batch_size)

                self.model.update(demo_batch, online_batch)

        if self.step_count >= self.min_replay_size and self.use_target_network \
                and self.step_count % self.target_update_steps == 0:
            self.model.update_target_network()

    def get_action(self, *args, **kwargs):
        return self.model.get_action(*args, **kwargs)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
