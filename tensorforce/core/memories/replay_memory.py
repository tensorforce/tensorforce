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

    def __init__(self, capacity, states, actions):
        self.states = dict()
        for state in states:
            name = state.get('name', 'state')
            self.states[name] = np.zeros((self.capacity,) + tuple(state['shape']), dtype=util.np_dtype(state.get('type')))
        self.actions = dict()
        for action in actions:
            name = action.get('name', 'action')
            dtype = util.np_dtype('float' if action['type'] == 'continuous' else 'int')
            self.actions[name] = np.zeros((self.capacity,), dtype=dtype)
        self.rewards = np.zeros((self.capacity,), dtype=util.np_dtype('float'))
        self.terminals = np.zeros((self.capacity,), dtype=util.np_dtype('bool'))
        self.internals = None

        self.capacity = capacity
        self.size = 0
        self.index = 0

    def add_experience(self, state, action, reward, terminal, internal):
        for name, state in state.items():
            self.states[name][self.index] = state
        for name, action in action.items():
            self.actions[name][self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal
        if self.internal is None and internal is not None:
            self.internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal in internal]
        for n in range(len(self.internals)):
            self.internals[n][self.index] = internal[n]

        if self.size < self.capacity:
            self.size += 1
        self.index = (self.index + 1) % self.capacity

    def get_batch(self, batch_size):
        end = (self.index - randrange(self.size - batch_size)) % self.capacity
        start = (end - batch_size) % self.capacity
        if start < end:
            indices = list(xrange(start, end))
        else:
            indices = list(xrange(start, self.capacity)) + list(xrange(0, end))

        batch = dict()
        for state in self.states:
            batch['states'][state] = self.states[state].take(indices)
        for action in self.actions:
            batch['actions'][action] = self.actions[action].take(indices)
        batch['rewards'] = self.rewards.take(indices)
        batch['terminals'] = self.rewards.take(indices)
        for n in range(len(self.internals)):
            batch['internals'].append(self.internals.take(indices))

        return batch
