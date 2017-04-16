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
Replay memory to store observations and sample
mini batches for training from.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np

from tensorforce.exceptions.tensorforce_exceptions import ArgumentMustBePositiveError
from tensorforce.util.experiment_util import global_seed


class ReplayMemory(object):
    def __init__(self,
                 memory_capacity,
                 state_shape,
                 action_shape,
                 state_type=np.float32,
                 action_type=np.int,
                 reward_type=np.float32,
                 deterministic_mode=False,
                 *args,
                 **kwargs):
        """
        Initializes a replay memory.

        :param memory_capacity: Memory size
        :param state_shape: Shape of state tensor
        :param state_type: Data type of state tensor
        :param action_shape: Shape of action tensor
        :param action_type: Data type of action tensor
        :param reward_type: Data type of reward function
        :param deterministic_mode: If true, global random number generation
        is controlled by passing the same seed to all generators, if false,
        no seed is used for sampling.
        """

        self.step_count = 0
        self.capacity = int(memory_capacity)
        self.size = 0

        # Explicitly set data types for every tensor to make for easier adjustments
        # if backend precision changes
        self.state_shape = state_shape
        self.state_type = state_type
        self.action_shape = action_shape
        self.action_type = action_type
        self.reward_type = reward_type

        # self batch shape
        self.states = np.zeros([self.capacity] + list(self.state_shape), dtype=self.state_type)
        self.actions = np.zeros([self.capacity] + list(self.action_shape), dtype=self.action_type)
        self.rewards = np.zeros([self.capacity], dtype=self.reward_type)
        self.terminals = np.zeros([self.capacity], dtype=bool)

        if deterministic_mode:
            self.random = global_seed()
        else:
            self.random = np.random.RandomState()

        # Index to control sampling
        self.top = 0

    def add_experience(self, state, action, reward, terminal):
        """
        Inserts an experience tuple to the replay_memory.

        :param state: State observed
        :param action: Action(s) taken
        :param reward: Reward seen after taking action
        :param terminal: Boolean whether episode ended
        :return:
        """

        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminals[self.top] = terminal

        if self.size < self.capacity:
            self.size += 1
        self.top = (self.top + 1) % self.capacity

    def sample_batch(self, batch_size):
        """
        Sample a mini batch of stored experiences.
        :param batch_size:
        :return: A Tensor containing experience tuples of length batch_size

        """
        batch_shape = [batch_size]

        if batch_size < 0:
            raise ArgumentMustBePositiveError('Batch size must be positive')

        batch_states = np.zeros(batch_shape + list(self.state_shape), dtype=self.state_type)
        batch_next_states = np.zeros(batch_shape + list(self.state_shape), dtype=self.state_type)

        batch_actions = np.zeros(batch_shape + list(self.action_shape), dtype=self.action_type)
        batch_rewards = np.zeros(batch_shape, dtype=self.reward_type)
        batch_terminals = np.zeros(batch_shape, dtype='bool')

        for i in xrange(batch_size):
            last_experience = self.top - 1 if self.top > 0 else self.size - 1
            index = last_experience

            # last added experience has no next state, avoid
            while index == last_experience:
                index = self.random.randint(self.size)

            if index == self.size:
                next_index = 0
            else:
                next_index = index + 1

            batch_states[i] = self.states.take(index, axis=0, mode='wrap')
            batch_actions[i] = self.actions.take(index, mode='wrap')
            batch_rewards[i] = self.rewards.take(index, mode='wrap')
            batch_terminals[i] = self.terminals.take(index, axis=0, mode='wrap')
            batch_next_states[i] = self.states.take(next_index, axis=0, mode='wrap')

        return dict(states=batch_states,
                    actions=batch_actions,
                    rewards=batch_rewards,
                    next_states=batch_next_states,
                    terminals=batch_terminals)
