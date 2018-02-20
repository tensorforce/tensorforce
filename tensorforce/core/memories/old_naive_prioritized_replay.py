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

from random import random, randrange
from six.moves import xrange
import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.core.memories import Memory


#TODO implement in TF
class NaivePrioritizedReplay(Memory):
    """
    Prioritised replay sampling based on loss per experience.
    """

    def __init__(self, states_spec, actions_spec, capacity, prioritization_weight=1.0):
        super(NaivePrioritizedReplay, self).__init__(states_spec=states_spec, actions_spec=actions_spec)
        self.capacity = capacity
        self.prioritization_weight = prioritization_weight
        self.internals_config = None
        # Stores (priority, observation) pairs in reverse priority order.
        self.observations = list()
        self.none_priority_index = 0
        self.batch_indices = None
        # Stores last observation until next_state value is known.
        self.last_observation = None

    def add_observation(self, states, internals, actions, terminal, reward):
        if self.internals_config is None and internals is not None:
            self.internals_config = [(internal.shape, internal.dtype) for internal in internals]

        if self.last_observation is not None:
            observation = self.last_observation + (states, internals)

            if len(self.observations) < self.capacity:
                self.observations.append((None, observation))
            elif self.none_priority_index > 0:
                priority, _ = self.observations.pop(self.none_priority_index - 1)
                self.observations.append((None, observation))
                self.none_priority_index -= 1
            else:
                raise TensorForceError("Memory contains only unseen observations.")

        self.last_observation = (states, internals, actions, terminal, reward)

    def get_batch(self, batch_size, next_states=False):
        """
        Samples a batch of the specified size according to priority.

        Args:
            batch_size: The batch size
            next_states: A boolean flag indicating whether 'next_states' values should be included

        Returns: A dict containing states, actions, rewards, terminals, internal states (and next states)

        """
        if batch_size > len(self.observations):
            raise TensorForceError("Batch size is larger than number of observations in memory.")

        states = {name: np.zeros((batch_size,) + tuple(state['shape']), dtype=util.np_dtype(state['type'])) for name, state in self.states_spec.items()}
        internals = [np.zeros((batch_size,) + shape, dtype) for shape, dtype in self.internals_config]
        actions = {name: np.zeros((batch_size,) + tuple(action['shape']), dtype=util.np_dtype(action['type'])) for name, action in self.actions_spec.items()}
        terminal = np.zeros((batch_size,), dtype=util.np_dtype('bool'))
        reward = np.zeros((batch_size,), dtype=util.np_dtype('float'))
        if next_states:
            next_states = {name: np.zeros((batch_size,) + tuple(state['shape']), dtype=util.np_dtype(state['type'])) for name, state in self.states_spec.items()}
            next_internals = [np.zeros((batch_size,) + shape, dtype) for shape, dtype in self.internals_config]

        self.batch_indices = list()
        not_sampled_index = self.none_priority_index
        sum_priorities = sum(priority for priority, _ in self.observations if priority is not None)
        for n in xrange(batch_size):
            if not_sampled_index < len(self.observations):
                _, observation = self.observations[not_sampled_index]
                index = not_sampled_index
                not_sampled_index += 1
            elif sum_priorities / self.capacity < util.epsilon:
                index = randrange(self.none_priority_index)
                while index in self.batch_indices:
                    index = randrange(self.none_priority_index)
                _, observation = self.observations[index]
            else:
                while True:
                    sample = random()
                    for index, (priority, observation) in enumerate(self.observations):
                        sample -= priority / sum_priorities
                        if sample < 0.0 or index >= self.none_priority_index:
                            break
                    if index not in self.batch_indices:
                        break

            for name, state in states.items():
                state[n] = observation[0][name]
            for k, internal in enumerate(internals):
                internal[n] = observation[1][k]
            for name, action in actions.items():
                action[n] = observation[2][name]
            terminal[n] = observation[3]
            reward[n] = observation[4]
            if next_states:
                for name, next_state in next_states.items():
                    next_state[n] = observation[5][name]
                for k, next_internal in enumerate(next_internals):
                    next_internal[n] = observation[6][k]
            self.batch_indices.append(index)

        if next_states:
            return dict(states=states, internals=internals, actions=actions, terminal=terminal, reward=reward, next_states=next_states, next_internals=next_internals)
        else:
            return dict(states=states, internals=internals, actions=actions, terminal=terminal, reward=reward)

    def update_batch(self, loss_per_instance):
        """
        Computes priorities according to loss.

        Args:
            loss_per_instance:

        """
        if self.batch_indices is None:
            raise TensorForceError("Need to call get_batch before each update_batch call.")
        # if len(loss_per_instance) != len(self.batch_indices):
        #     raise TensorForceError("For all instances a loss value has to be provided.")

        updated = list()
        for index, loss in zip(self.batch_indices, loss_per_instance):
            priority, observation = self.observations[index]
            updated.append((loss ** self.prioritization_weight, observation))
        for index in sorted(self.batch_indices, reverse=True):
            priority, _ = self.observations.pop(index)
            self.none_priority_index -= (priority is not None)
        self.batch_indices = None
        updated = sorted(updated, key=(lambda x: x[0]))

        update_priority, update_observation = updated.pop()
        index = -1
        for priority, _ in iter(self.observations):
            index += 1
            if index == self.none_priority_index:
                break
            if update_priority < priority:
                continue
            self.observations.insert(index, (update_priority, update_observation))
            index += 1
            self.none_priority_index += 1
            if not updated:
                break
            update_priority, update_observation = updated.pop()
        else:
            self.observations.insert(index, (update_priority, update_observation))
            self.none_priority_index += 1
        while updated:
            self.observations.insert(index, updated.pop())
            self.none_priority_index += 1
