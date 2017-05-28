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
Replay memory to store observations and sample mini batches for training from.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from random import randrange
from six.moves import xrange
import numpy as np

from tensorforce import util
from tensorforce.core.memories import Memory


class ReplayMemory(Memory):

    def __init__(self, capacity, states_config, actions_config):
        capacity = int(capacity)
        self.states = dict()
        for name, state in states_config:
            self.states[name] = np.zeros((capacity,) + tuple(state.shape), dtype=util.np_dtype(state.type))
        self.actions = dict()
        for name, action in actions_config:
            dtype = util.np_dtype('float' if action.continuous else 'int')
            self.actions[name] = np.zeros((capacity,), dtype=dtype)
        self.rewards = np.zeros((capacity,), dtype=util.np_dtype('float'))
        self.terminals = np.zeros((capacity,), dtype=util.np_dtype('bool'))
        self.internals = None

        self.capacity = capacity
        self.size = 0
        self.index = 0

    def add_experience(self, state, action, reward, terminal, internal):
        if self.internals is None and internal is not None:
            self.internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal in internal]

        for name, state in state.items():
            self.states[name][self.index] = state
        for name, action in action.items():
            self.actions[name][self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal
        for n, internal in enumerate(internal):
            self.internals[n][self.index] = internal

        if self.size < self.capacity:
            self.size += 1
        self.index = (self.index + 1) % self.capacity

    def get_batch(self, batch_size):
        """
        Samples a batch of the specified size by selecting a random start/end point and returning
        the contained sequence (as opposed to sampling each state separately).
        
        Args:
            batch_size: Length of the sampled sequence.

        Returns: A dict containing states, rewards, terminals and internal states

        """
        end = (self.index - randrange(self.size - batch_size)) % self.capacity
        start = (end - batch_size) % self.capacity
        if start < end:
            indices = list(xrange(start, end))
        else:
            indices = list(xrange(start, self.capacity)) + list(xrange(0, end))

        return dict(
            states={name: state.take(indices, axis=0) for name, state in self.states.items()},
            actions={name: action.take(indices) for name, action in self.actions.items()},
            rewards=self.rewards.take(indices),
            terminals=self.terminals.take(indices),
            internals=[internal.take(indices, axis=0) for internal in self.internals]
        )
