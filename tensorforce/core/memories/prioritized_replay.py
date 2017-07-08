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

from random import random
from six.moves import xrange
import numpy as np

from tensorforce import util
from tensorforce.core.memories import Memory


class PrioritizedReplay(Memory):

    def __init__(self, capacity, states_config, actions_config, prioritization_weight=1.0):
        self.capacity = capacity
        self.prioritization_weight = prioritization_weight

        self.state_spec = {name: (tuple(state.shape), util.np_dtype(state.type)) for name, state in states_config}
        self.action_spec = {name: util.np_dtype('float' if action.continuous else 'int') for name, action in actions_config}
        self.internal_spec = None
        self.observations = list()

        self.sum_priorities = 0.0
        self.positive_priority_index = -1
        self.batch_indices = list()

    def add_observation(self, state, action, reward, terminal, internal):
        if self.internal_spec is None and internal is not None:
            self.internal_spec = [(i.shape, i.dtype) for i in internal]

        observation = (state, action, reward, terminal, internal)
        if len(self.observations) < self.capacity:
            self.observations.append((0.0, observation))
        else:
            priority, _ = self.observations.pop(self.positive_priority_index)
            self.observations.append((0.0, observation))
            self.sum_priorities -= priority
            self.positive_priority_index -= 1

    def get_batch(self, batch_size):
        """
        Samples a batch of the specified size by selecting a random start/end point and returning
        the contained sequence (as opposed to sampling each state separately).

        Args:
            batch_size: Length of the sampled sequence.

        Returns: A dict containing states, rewards, terminals and internal states

        """
        assert not self.batch_indices

        states = {name: np.zeros((batch_size,) + tuple(shape), dtype=dtype) for name, (shape, dtype) in self.state_spec.items()}
        actions = {name: np.zeros((batch_size,), dtype=dtype) for name, dtype in self.action_spec.items()}
        rewards = np.zeros((batch_size,), dtype=util.np_dtype('float'))
        terminals = np.zeros((batch_size,), dtype=util.np_dtype('bool'))
        internals = [np.zeros((batch_size,) + shape, dtype) for shape, dtype in self.internal_spec]

        zero_priority_index = self.positive_priority_index + 1
        for n in xrange(batch_size):
            if zero_priority_index < len(self.observations):
                _, observation = self.observations[zero_priority_index]
                index = zero_priority_index
                zero_priority_index += 1
            else:
                while True:
                    sample = random()
                    for index, (priority, observation) in enumerate(self.observations):
                        sample -= priority / self.sum_priorities
                        if sample < 0.0:
                            break
                    if index not in self.batch_indices:
                        break

            for name, state in states.items():
                state[n] = observation[0][name]
            for name, action in actions.items():
                action[n] = observation[1][name]
            rewards[n] = observation[2]
            terminals[n] = observation[3]
            for k, internal in enumerate(internals):
                internal[n] = observation[4][k]
            self.batch_indices.append(index)

        return dict(states=states, actions=actions, rewards=rewards, terminals=terminals, internals=internals)

    def update_batch(self, loss_per_instance):
        assert self.batch_indices
        self.batch_indices = self.batch_indices[:len(loss_per_instance)]

        updated = list()
        for index, loss in zip(self.batch_indices, loss_per_instance):
            priority, observation = self.observations[index]
            self.sum_priorities -= priority
            if priority > 0.0:
                self.positive_priority_index -= 1
            updated.append((loss ** self.prioritization_weight, observation))
        for index in sorted(self.batch_indices, reverse=True):
            self.observations.pop(index)
        self.batch_indices = list()
        updated = sorted(updated, key=(lambda x: x[0]))

        update_priority, update_observation = updated.pop()
        for n, (priority, _) in enumerate(self.observations):
            if update_priority <= priority:
                continue
            self.observations.insert(n, (update_priority, update_observation))
            self.sum_priorities += update_priority
            if update_priority > 0.0:
                self.positive_priority_index += 1
            if not updated:
                break
            update_priority, update_observation = updated.pop()
        else:
            self.observations.append((update_priority, update_observation))
        while updated:
            self.observations.append(updated.pop())
