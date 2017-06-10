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

from tensorforce.core import Agent
from tensorforce.core.memories import ReplayMemory
from tensorforce.models import DQFDModel


class DQFDAgent(Agent):

    name = 'DQFDAgent'
    model = DQFDModel

    default_config = {
        "expert_sampling_ratio": 0.01,
        'batch_size': 32,
        'update_rate': 0.25,
        'target_network_update_rate': 0.0001,
        'min_replay_size': 5e4,
        'deterministic_mode': False,
        'use_target_network': False,
        'update_repeat': 1
    }

    def __init__(self, config):
        """
        
        :param config: 
        :param scope: 
        """
        config.default(DQFDAgent.default_config)
        super(DQFDAgent, self).__init__(config)

        # This is the online memory
        self.replay_memory = ReplayMemory(capacity=config.memory_capacity, states_config=config.states, actions_config=config.actions)

        # This is the demonstration memory that we will fill with observations before starting
        # the main training loop
        # TODO we might want different sizes for these memories -> add config param
        self.demo_memory = ReplayMemory(capacity=config.memory_capacity, states_config=config.states, actions_config=config.actions)
        self.step_count = 0

        # Called p in paper, controls ratio of expert vs online training samples
        self.expert_sampling_ratio = config.expert_sampling_ratio

        self.update_repeat = config.update_repeat
        self.batch_size = config.batch_size

        # p = n_demo / (n_demo + n_replay) => n_demo  = p * n_replay / (1 - p)
        self.demo_batch_size = int(self.expert_sampling_ratio * self.batch_size / \
                               (1.0 - self.expert_sampling_ratio))
        self.update_steps = int(round(1 / config.update_rate))
        self.use_target_network = config.use_target_network

        if self.use_target_network:
            self.target_update_steps = int(round(1 / config.target_network_update_rate))

        self.min_replay_size = config.min_replay_size

    def add_demo_observation(self, state, action, reward, terminal):
        """Adds observations to demo memory. 

        Args:
            state: 
            action: 
            reward: 
            terminal: 

        Returns:

        """
        if self.unique_state:
            state = dict(state=state)
        if self.unique_action:
            action = dict(action=action)

        self.demo_memory.add_experience(state, action, reward, terminal, internal=self.internals)

    def pre_train(self, steps=1):
        """Computes pretrain updates.
        
        Args:
            steps: Number of updates to execute.

        Returns:

        """
        for _ in xrange(steps):
            # Sample from demo memory
            batch = self.demo_memory.get_batch(self.batch_size)

            # Update using both double Q-learning and supervised double_q_loss
            self.model.pre_train_update(batch)

    def observe(self, state, action, reward, terminal):
        """Adds observations, updates via sampling from memories according to update rate.
        DQFD samples from the online replay memory and the demo memory with
        the fractions controlled by a hyper parameter p called 'expert sampling ratio.
        
        Args:
            state: 
            action: 
            reward: 
            terminal: 

        Returns:

        """
        if self.unique_state:
            state = dict(state=state)
        if self.unique_action:
            action = dict(action=action)

        self.replay_memory.add_experience(state, action, reward, terminal, internal=self.internals)

        self.step_count += 1

        if self.step_count >= self.min_replay_size and self.step_count % self.update_steps == 0:
            for _ in xrange(self.update_repeat):
                # Sample batches according to expert sampling ratio
                # In the paper, p is given as p = n_demo / (n_replay + n_demo)
                demo_batch = self.demo_memory.get_batch(self.demo_batch_size)
                online_batch = self.replay_memory.get_batch(self.batch_size)

                self.model.pre_train_update(batch=demo_batch)
                self.model.update(batch=online_batch)

        if self.step_count >= self.min_replay_size and self.use_target_network \
                and self.step_count % self.target_update_steps == 0:
            self.model.update_target_network()

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
