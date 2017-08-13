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


class Replay(Memory):

    def __init__(self, capacity, states_config, actions_config, random_sampling=False):
        super(Replay, self).__init__(capacity, states_config, actions_config)
        self.states = {name: np.zeros((capacity,) + tuple(state.shape), dtype=util.np_dtype(state.type)) for name, state in states_config}
        self.actions = {name: np.zeros((capacity,) + tuple(action.shape), dtype=util.np_dtype('float' if action.continuous else 'int')) for name, action in actions_config}
        self.rewards = np.zeros((capacity,), dtype=util.np_dtype('float'))
        self.terminals = np.zeros((capacity,), dtype=util.np_dtype('bool'))
        self.internals = None
        self.size = 0
        self.index = 0
        self.random_sampling = random_sampling

    def add_observation(self, state, action, reward, terminal, internal):
        if self.internals is None and internal is not None:
            self.internals = [np.zeros((self.capacity,) + i.shape, i.dtype) for i in internal]

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

    def get_batch(self, batch_size, next_states=False):
        """
        Samples a batch of the specified size by selecting a random start/end point and returning
        the contained sequence or random indices depending on the field 'random_sampling'
        
        Args:
            batch_size: The batch size
            next_states: A boolean flag indicating whether 'next_states' values should be included

        Returns: A dict containing states, actions, rewards, terminals, internal states (and next states)

        """
        if self.random_sampling:
            indices = np.random.randint(self.size, size=batch_size)
            states = {name: state.take(indices, axis=0) for name, state in self.states.items()}
            actions = {name: action.take(indices, axis=0) for name, action in self.actions.items()}
            rewards = self.rewards.take(indices)
            terminals = self.terminals.take(indices)
            internals = [internal.take(indices, axis=0) for internal in self.internals]
            if next_states:
                indices = (indices + 1) % self.capacity
                next_states = {name: state.take(indices, axis=0) for name, state in self.states.items()}
                next_internals = [internal.take(indices, axis=0) for internal in self.internals]

        else:
            end = (self.index - randrange(self.size - batch_size + 1)) % self.capacity
            start = (end - batch_size) % self.capacity

            if start < end:
                states = {name: state[start:end] for name, state in self.states.items()}
                actions = {name: action[start:end] for name, action in self.actions.items()}
                rewards = self.rewards[start:end]
                terminals = self.terminals[start:end]
                internals = [internal[start:end] for internal in self.internals]
                if next_states:
                    next_states = {name: state[start + 1: end + 1] for name, state in self.states.items()}
                    next_internals = [internal[start + 1: end + 1] for internal in self.internals]

            else:
                states = {name: np.concatenate((state[start:], state[:end])) for name, state in self.states.items()}
                actions = {name: np.concatenate((action[start:], action[:end])) for name, action in self.actions.items()}
                rewards = np.concatenate((self.rewards[start:], self.rewards[:end]))
                terminals = np.concatenate((self.terminals[start:], self.terminals[:end]))
                internals = [np.concatenate((internal[start:], internal[:end])) for internal in self.internals]
                if next_states:
                    next_states = {name: np.concatenate((state[start + 1:], state[:end + 1])) for name, state in self.states.items()}
                    next_internals = [np.concatenate((internal[start + 1:], internal[:end + 1])) for internal in self.internals]

        batch = dict(states=states, actions=actions, rewards=rewards, terminals=terminals, internals=internals)
        if next_states:
            batch['next_states'] = next_states
            batch['next_internals'] = next_internals
        return batch

    def update_batch(self, loss_per_instance):
        pass

    def set_memory(self, states, actions, rewards, terminals, internals):
        self.size = len(rewards)

        if len(rewards) == self.capacity:
            # Assign directly if capacity matches size.
            for name, state in states.items():
                self.states[name] = np.asarray(state)
            for name, action in actions.items():
                self.actions[name] = np.asarray(action)
            self.rewards = np.asarray(rewards)
            self.terminals = np.asarray(terminals)
            self.internals = [np.asarray(internal) for internal in internals]

        else:
            # Otherwise partial assignment
            for name, state in states.items():
                self.states[name][:len(state)] = state
            for name, action in actions.items():
                self.actions[name][:len(action)] = action
            self.rewards[:len(rewards)] = rewards
            self.terminals[:len(terminals)] = terminals
            if self.internals is None and internals is not None:
                self.internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal in internals]
            for n, internal in enumerate(internals):
                self.internals[n][:len(internal)] = internal
