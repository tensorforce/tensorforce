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

import tensorflow as tf

from tensorforce import util
from tensorforce.core.memories import Memory


class Queue(Memory):
    """
    Base class for memories organized as a queue (FIFO).
    """

    def __init__(self, states, internals, actions, include_next_states, capacity, scope='queue', summary_labels=None):
        """
        Queue memory.

        Args:
            states: States specifiction.
            internals: Internal states specification.
            actions: Actions specification.
            include_next_states: Include subsequent state if true.
            capacity: Memory capacity.
        """
        super(Queue, self).__init__(
            states=states,
            internals=internals,
            actions=actions,
            include_next_states=include_next_states,
            scope=scope,
            summary_labels=summary_labels
        )
        self.capacity = capacity

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if not registered:
                assert not kwargs.get('trainable', False)
                self.variables[name] = variable
            return variable

        self.retrieve_indices = tf.make_template(
            name_=(scope + '/retrieve_indices'),
            func_=self.tf_retrieve_indices,
            custom_getter_=custom_getter
        )

    def tf_initialize(self):
        # States
        self.states_memory = dict()
        for name, state in self.states_spec.items():
            self.states_memory[name] = tf.get_variable(
                name=('state-' + name),
                shape=(self.capacity,) + tuple(state['shape']),
                dtype=util.tf_dtype(state['type']),
                trainable=False
            )

        # Internals
        self.internals_memory = dict()
        for name, internal in self.internals_spec.items():
            self.internals_memory[name] = tf.get_variable(
                name=('internal-' + name),
                shape=(self.capacity,) + tuple(internal['shape']),
                dtype=util.tf_dtype(internal['type']),
                trainable=False
            )

        # Actions
        self.actions_memory = dict()
        for name, action in self.actions_spec.items():
            self.actions_memory[name] = tf.get_variable(
                name=('action-' + name),
                shape=(self.capacity,) + tuple(action['shape']),
                dtype=util.tf_dtype(action['type']),
                trainable=False
            )

        # Terminal
        self.terminal_memory = tf.get_variable(
            name='terminal',
            shape=(self.capacity,),
            dtype=util.tf_dtype('bool'),
            initializer=tf.constant_initializer(
                value=tuple(n == self.capacity - 1 for n in range(self.capacity)),
                dtype=util.tf_dtype('bool')
            ),
            trainable=False
        )

        # Reward
        self.reward_memory = tf.get_variable(
            name='reward',
            shape=(self.capacity,),
            dtype=util.tf_dtype('float'),
            trainable=False
        )

        # Memory index
        self.memory_index = tf.get_variable(
            name='memory-index',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

        # Episode indices
        self.episode_indices = tf.get_variable(
            name='episode-indices',
            shape=(self.capacity + 1,),
            dtype=util.tf_dtype('int'),
            initializer=tf.constant_initializer(value=(self.capacity - 1), dtype=util.tf_dtype('int')),
            trainable=False
        )

        # Episodes index
        self.episode_count = tf.get_variable(
            name='episode-count',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

    def tf_store(self, states, internals, actions, terminal, reward):
        # Memory indices to overwrite.
        num_instances = tf.shape(input=terminal)[0]
        indices = tf.range(start=self.memory_index, limit=(self.memory_index + num_instances)) % self.capacity

        # Remove episode indices.
        num_episodes = tf.count_nonzero(
            input_tensor=tf.gather(params=self.terminal_memory, indices=indices),
            axis=0,
            dtype=util.tf_dtype('int')
        )
        num_episodes = tf.minimum(x=num_episodes, y=self.episode_count)
        assignment = tf.assign(
            ref=self.episode_indices[:self.episode_count + 1 - num_episodes],
            value=self.episode_indices[num_episodes: self.episode_count + 1]
        )

        # Decrement episode count.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign_sub(ref=self.episode_count, value=num_episodes)

        # Assign new observations.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignments = list()
            for name, state in states.items():
                assignments.append(tf.scatter_update(
                    ref=self.states_memory[name],
                    indices=indices,
                    updates=state
                ))
            for name, internal in internals.items():
                assignments.append(tf.scatter_update(
                    ref=self.internals_memory[name],
                    indices=indices,
                    updates=internal
                ))
            for name, action in actions.items():
                assignments.append(tf.scatter_update(
                    ref=self.actions_memory[name],
                    indices=indices,
                    updates=action
                ))
            assignments.append(tf.scatter_update(ref=self.terminal_memory, indices=indices, updates=terminal))
            assignments.append(tf.scatter_update(ref=self.reward_memory, indices=indices, updates=reward))

        # Increment memory index.
        with tf.control_dependencies(control_inputs=assignments):
            assignment = tf.assign(ref=self.memory_index, value=((self.memory_index + num_instances) % self.capacity))

        # Add episode indices.
        with tf.control_dependencies(control_inputs=(assignment,)):
            num_episodes = tf.count_nonzero(input_tensor=terminal, axis=0, dtype=util.tf_dtype('int'))
            assignment = tf.assign(
                ref=self.episode_indices[self.episode_count + 1: self.episode_count + 1 + num_episodes],
                value=tf.boolean_mask(tensor=indices, mask=terminal)
            )

        # Increment episode count.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign_add(ref=self.episode_count, value=num_episodes)

        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.no_op()

    def tf_retrieve_indices(self, indices):
        """
        Fetches experiences for given indices.

        Args:
            indices: Index tensor

        Returns: Batch of experiences
        """
        states = dict()
        for name, state_memory in self.states_memory.items():
            states[name] = tf.gather(params=state_memory, indices=indices)

        internals = dict()
        for name, internal_memory in self.internals_memory.items():
            internals[name] = tf.gather(params=internal_memory, indices=indices)

        actions = dict()
        for name, action_memory in self.actions_memory.items():
            actions[name] = tf.gather(params=action_memory, indices=indices)

        terminal = tf.gather(params=self.terminal_memory, indices=indices)
        reward = tf.gather(params=self.reward_memory, indices=indices)

        if self.include_next_states:
            assert util.rank(indices) == 1
            next_indices = (indices + 1) % self.capacity

            next_states = dict()
            for name, state_memory in self.states_memory.items():
                next_states[name] = tf.gather(params=state_memory, indices=next_indices)

            next_internals = dict()
            for name, internal_memory in self.internals_memory.items():
                next_internals[name] = tf.gather(params=internal_memory, indices=next_indices)

            return dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                next_states=next_states,
                next_internals=next_internals
            )
        else:
            return dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )
