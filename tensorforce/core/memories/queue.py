# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from tensorforce import util
from tensorforce.core.memories import Memory


class Queue(Memory):
    """
    Base class for memories organized as a queue (FIFO).
    """

    def __init__(
        self, name, states_spec, internals_spec, actions_spec, include_next_states, capacity,
        summary_labels=None
    ):
        """
        Queue memory.

        Args:
            capacity: Memory capacity.
        """
        super().__init__(
            name=name, states_spec=states_spec, internals_spec=internals_spec,
            actions_spec=actions_spec, include_next_states=include_next_states,
            summary_labels=summary_labels
        )

        self.capacity = capacity

    def tf_initialize(self):
        super().tf_initialize()

        self.memories = OrderedDict()

        # State memories
        for name, state_spec in self.states_spec.items():
            self.memories[name] = self.add_variable(
                name=(name + '-memory'), dtype=state_spec['type'],
                shape=(self.capacity,) + tuple(state_spec['shape']), is_trainable=False
            )

        # Internal memories
        for name, internal_spec in self.internals_spec.items():
            self.memories[name] = self.add_variable(
                name=(name + '-memory'), dtype=internal_spec['type'],
                shape=(self.capacity,) + tuple(internal_spec['shape']), is_trainable=False
            )

        # Action memories
        for name, action_spec in self.actions_spec.items():
            self.memories[name] = self.add_variable(
                name=(name + '-memory'), dtype=action_spec['type'],
                shape=(self.capacity,) + tuple(action_spec['shape']), is_trainable=False
            )

        # Terminal memory (initialization to agree with terminal_indices)
        initializer = np.zeros(shape=(self.capacity,), dtype=util.np_dtype(dtype='bool'))
        initializer[-1] = True
        self.memories['terminal'] = self.add_variable(
            name='terminal-memory', dtype='bool', shape=(self.capacity,), is_trainable=False,
            initializer=initializer
        )

        # Reward memory
        self.memories['reward'] = self.add_variable(
            name='reward-memory', dtype='float', shape=(self.capacity,), is_trainable=False
        )

        # Memory index (next index to write to)
        self.memory_index = self.add_variable(
            name='memory-index', dtype='long', shape=(), is_trainable=False, initializer='zeros'
        )

        # Terminal indices (oldest episode terminals first, initial only terminal is last index)
        initializer = np.zeros(shape=(self.capacity + 1,), dtype=util.np_dtype(dtype='long'))
        initializer[0] = self.capacity - 1
        self.terminal_indices = self.add_variable(
            name='terminal-indices', dtype='long', shape=(self.capacity + 1,), is_trainable=False,
            initializer=initializer
        )

        # Episode count
        self.episode_count = self.add_variable(
            name='episode-count', dtype='long', shape=(), is_trainable=False, initializer='zeros'
        )

    def tf_store(self, states, internals, actions, terminal, reward):
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        # Check whether instances fit into memory
        num_timesteps = tf.shape(input=terminal, out_type=util.tf_dtype(dtype='long'))[0]
        assertion = tf.debugging.assert_less_equal(x=num_timesteps, y=capacity)

        # Memory indices to overwrite
        with tf.control_dependencies(control_inputs=(assertion,)):
            indices = tf.range(start=self.memory_index, limit=(self.memory_index + num_timesteps))
            indices = tf.mod(x=indices, y=capacity)

            # Count number of overwritten episodes
            num_episodes = tf.count_nonzero(
                input_tensor=tf.gather(params=self.memories['terminal'], indices=indices), axis=0,
                dtype=util.tf_dtype(dtype='long')
            )

            # Shift remaining terminal indices accordingly
            limit_index = self.episode_count + one
            assignment = tf.assign(
                ref=self.terminal_indices[:limit_index - num_episodes],
                value=self.terminal_indices[num_episodes: limit_index]
            )

        # Decrement episode count accordingly
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = self.episode_count.assign_sub(delta=num_episodes, read_value=False)

        # Write new observations
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignments = list()
            for name, state in states.items():
                assignments.append(
                    tf.scatter_update(ref=self.memories[name], indices=indices, updates=state)
                )
            for name, internal in internals.items():
                assignments.append(
                    tf.scatter_update(ref=self.memories[name], indices=indices, updates=internal)
                )
            for name, action in actions.items():
                assignments.append(
                    tf.scatter_update(ref=self.memories[name], indices=indices, updates=action)
                )
            assignments.append(
                tf.scatter_update(ref=self.memories['terminal'], indices=indices, updates=terminal)
            )
            assignments.append(
                tf.scatter_update(ref=self.memories['reward'], indices=indices, updates=reward)
            )

        # Increment memory index
        with tf.control_dependencies(control_inputs=assignments):
            new_memory_index = tf.mod(x=(self.memory_index + num_timesteps), y=capacity)
            assignment = self.memory_index.assign(value=new_memory_index)

        # Count number of new episodes
        with tf.control_dependencies(control_inputs=(assignment,)):
            num_new_episodes = tf.count_nonzero(
                input_tensor=terminal, axis=0, dtype=util.tf_dtype(dtype='long')
            )

            # Write new terminal indices
            limit_index = self.episode_count + one
            assignment = tf.assign(
                ref=self.terminal_indices[limit_index: limit_index + num_new_episodes],
                value=tf.boolean_mask(tensor=indices, mask=terminal)
            )

        # Increment episode count accordingly
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = self.episode_count.assign_add(delta=num_new_episodes, read_value=False)

        with tf.control_dependencies(control_inputs=(assignment,)):
            return util.no_operation()

    def tf_retrieve_indices(self, indices):
        """
        Fetches experiences for given indices.

        Args:
            indices: Index tensor

        Returns: Batch of experiences
        """
        assert util.rank(x=indices) == 1
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        if self.include_next_states:
            next_indices = (indices + one) % capacity
            next_terminal = tf.gather(params=self.memories['terminal'], indices=next_indices)
            indices = tf.boolean_mask(tensor=indices, mask=tf.math.logical_not(x=next_terminal))

        states = OrderedDict()
        for name in self.states_spec:
            states[name] = tf.gather(params=self.memories[name], indices=indices)

        internals = OrderedDict()
        for name in self.internals_spec:
            internals[name] = tf.gather(params=self.memories[name], indices=indices)

        actions = OrderedDict()
        for name in self.actions_spec:
            actions[name] = tf.gather(params=self.memories[name], indices=indices)

        terminal = tf.gather(params=self.memories['terminal'], indices=indices)

        reward = tf.gather(params=self.memories['reward'], indices=indices)

        if self.include_next_states:
            next_indices = (indices + one) % capacity

            next_states = OrderedDict()
            for name in self.states_spec:
                next_states[name] = tf.gather(params=self.memories[name], indices=next_indices)

            next_internals = OrderedDict()
            for name in self.internals_spec:
                next_internals[name] = tf.gather(params=self.memories[name], indices=next_indices)

            return dict(
                states=states, internals=internals, actions=actions, terminal=terminal,
                reward=reward, next_states=next_states, next_internals=next_internals
            )

        else:
            return dict(
                states=states, internals=internals, actions=actions, terminal=terminal,
                reward=reward
            )
