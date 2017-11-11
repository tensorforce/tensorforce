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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from random import randrange
import numpy as np

from tensorforce import util
from tensorforce.core.memories import Memory


class Replay(Memory):
    """
    Replay memory to store observations and sample mini batches for training from.
    """

    def __init__(self, states_spec, actions_spec, capacity, random_sampling=True):
        super(Replay, self).__init__(states_spec=states_spec, actions_spec=actions_spec)
        self.capacity = capacity
        self.states = {name: np.zeros((capacity,) + tuple(state['shape']), dtype=util.np_dtype(state['type'])) for name, state in states_spec.items()}
        self.internals = None
        self.actions = {name: np.zeros((capacity,) + tuple(action['shape']), dtype=util.np_dtype(action['type'])) for name, action in actions_spec.items()}
        self.terminal = np.zeros((capacity,), dtype=util.np_dtype('bool'))
        self.reward = np.zeros((capacity,), dtype=util.np_dtype('float'))

        self.size = 0
        self.index = 0
        self.random_sampling = random_sampling

    def add_observation(self, states, internals, actions, terminal, reward):
        if self.internals is None and internals is not None:
            self.internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal in internals]

        for name, state in states.items():
            self.states[name][self.index] = state
        for n, internal in enumerate(internals):
            self.internals[n][self.index] = internal
        for name, action in actions.items():
            self.actions[name][self.index] = action
        self.reward[self.index] = reward
        self.terminal[self.index] = terminal

        if self.size < self.capacity:
            self.size += 1
        self.index = (self.index + 1) % self.capacity

    def get_batch(self, batch_size, next_states=False):
        """
        Samples a batch of the specified size by selecting a random start/end point and returning
        the contained sequence or random indices depending on the field 'random_sampling'.
        
        Args:
            batch_size: The batch size
            next_states: A boolean flag indicating whether 'next_states' values should be included

        Returns: A dict containing states, actions, rewards, terminals, internal states (and next states)

        """
        if self.random_sampling:
            if next_states:
                indices = np.random.randint(self.size - 1, size=batch_size)
            else:
                indices = np.random.randint(self.size, size=batch_size)

            states = {name: state.take(indices, axis=0) for name, state in self.states.items()}
            internals = [internal.take(indices, axis=0) for internal in self.internals]
            actions = {name: action.take(indices, axis=0) for name, action in self.actions.items()}
            terminal = self.terminal.take(indices)
            reward = self.reward.take(indices)
            if next_states:
                indices = (indices + 1) % self.capacity
                next_states = {name: state.take(indices, axis=0) for name, state in self.states.items()}
                next_internals = [internal.take(indices, axis=0) for internal in self.internals]

        else:
            if next_states:
                end = (self.index - 1 - randrange(self.size - batch_size + 1)) % self.capacity
                start = (end - batch_size) % self.capacity
            else:
                end = (self.index - randrange(self.size - batch_size + 1)) % self.capacity
                start = (end - batch_size) % self.capacity

            if start < end:
                states = {name: state[start:end] for name, state in self.states.items()}
                internals = [internal[start:end] for internal in self.internals]
                actions = {name: action[start:end] for name, action in self.actions.items()}
                terminal = self.terminal[start:end]
                reward = self.reward[start:end]
                if next_states:
                    next_states = {name: state[start + 1: end + 1] for name, state in self.states.items()}
                    next_internals = [internal[start + 1: end + 1] for internal in self.internals]

            else:
                states = {name: np.concatenate((state[start:], state[:end])) for name, state in self.states.items()}
                internals = [np.concatenate((internal[start:], internal[:end])) for internal in self.internals]
                actions = {name: np.concatenate((action[start:], action[:end])) for name, action in self.actions.items()}
                terminal = np.concatenate((self.terminal[start:], self.terminal[:end]))
                reward = np.concatenate((self.reward[start:], self.reward[:end]))
                if next_states:
                    next_states = {name: np.concatenate((state[start + 1:], state[:end + 1])) for name, state in self.states.items()}
                    next_internals = [np.concatenate((internal[start + 1:], internal[:end + 1])) for internal in self.internals]

        batch = dict(states=states, internals=internals, actions=actions, terminal=terminal, reward=reward)
        if next_states:
            batch['next_states'] = next_states
            batch['next_internals'] = next_internals
        return batch

    def update_batch(self, loss_per_instance):
        pass

    def set_memory(self, states, internals, actions, terminal, reward):
        """
        Convenience function to set whole batches as memory content to bypass
        calling the insert function for every single experience.

        Args:
            states:
            internals:
            actions:
            terminal:
            reward:

        Returns:

        """
        self.size = len(terminal)

        if len(terminal) == self.capacity:
            # Assign directly if capacity matches size.
            for name, state in states.items():
                self.states[name] = np.asarray(state)
            self.internals = [np.asarray(internal) for internal in internals]
            for name, action in actions.items():
                self.actions[name] = np.asarray(action)
            self.terminal = np.asarray(terminal)
            self.reward = np.asarray(reward)

        else:
            # Otherwise partial assignment.
            if self.internals is None and internals is not None:
                self.internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal
                                  in internals]

            for name, state in states.items():
                self.states[name][:len(state)] = state
            for n, internal in enumerate(internals):
                self.internals[n][:len(internal)] = internal
            for name, action in actions.items():
                self.actions[name][:len(action)] = action
            self.terminal[:len(terminal)] = terminal
            self.reward[:len(reward)] = reward
