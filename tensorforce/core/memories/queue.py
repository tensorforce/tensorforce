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
            capacity: Memory capacity.
        """
        self.capacity = capacity
        self.scope = scope

        # Pieces of the records are stored in different tensors:
        self.states_memory = dict()  # keys=state space components
        self.internals_memory = dict()  # keys=internal state components
        self.actions_memory = dict()  # keys=action space components
        self.terminal_memory = None  # 1D tensor
        self.reward_memory = None  # 1D tensor
        self.memory_index = None  # 0D (int) tensor (points to the next record to be overwritten)
        self.episode_indices = None  # 1D tensor of indexes where episodes start.
        self.episode_count = None  # 0D (int) tensor: How many episodes do we have stored?

        self.retrieve_indices = None

        super(Queue, self).__init__(
            states=states,
            internals=internals,
            actions=actions,
            include_next_states=include_next_states,
            scope=scope,
            summary_labels=summary_labels
        )

    def setup_template_funcs(self, custom_getter=None):
        custom_getter = super(Queue, self).setup_template_funcs(custom_getter=custom_getter)

        self.retrieve_indices = tf.make_template(
            name_=(self.scope + '/retrieve_indices'),
            func_=self.tf_retrieve_indices,
            custom_getter_=custom_getter
        )

    def tf_initialize(self):
        # States
        for name in sorted(self.states_spec):
            state = self.states_spec[name]
            self.states_memory[name] = tf.get_variable(
                name=('state-' + name),
                shape=(self.capacity,) + tuple(state['shape']),
                dtype=util.tf_dtype(state['type']),
                trainable=False
            )

        # Internals
        for name in sorted(self.internals_spec):
            internal = self.internals_spec[name]
            self.internals_memory[name] = tf.get_variable(
                name=('internal-' + name),
                shape=(self.capacity,) + tuple(internal['shape']),
                dtype=util.tf_dtype(internal['type']),
                trainable=False
            )

        # Actions
        for name in sorted(self.actions_spec):
            action = self.actions_spec[name]
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
                value=False,
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
        with tf.control_dependencies([tf.assert_less_equal(num_instances, self.capacity)]):
            indices = tf.range(self.memory_index, self.memory_index + num_instances) % self.capacity

        # Remove episode indices.
        num_episodes = tf.count_nonzero(
            input_tensor=tf.gather(params=self.terminal_memory, indices=indices),
            axis=0,
            dtype=util.tf_dtype('int')
        )
        num_episodes = tf.minimum(x=num_episodes, y=self.episode_count)
        assignment = tf.assign(
            ref=self.episode_indices[:self.episode_count - num_episodes],
            value=self.episode_indices[num_episodes: self.episode_count]
        )

        # Decrement episode count.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign_sub(ref=self.episode_count, value=num_episodes)

        # Assign new observations.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignments = list()
            for name in sorted(states):
                assignments.append(tf.scatter_update(
                    ref=self.states_memory[name],
                    indices=indices,
                    updates=states[name]
                ))
            for name in sorted(internals):
                assignments.append(tf.scatter_update(
                    ref=self.internals_memory[name],
                    indices=indices,
                    updates=internals[name]
                ))
            for name in sorted(actions):
                assignments.append(tf.scatter_update(
                    ref=self.actions_memory[name],
                    indices=indices,
                    updates=actions[name]
                ))
            assignments.append(tf.scatter_update(ref=self.terminal_memory, indices=indices, updates=terminal))
            assignments.append(tf.scatter_update(ref=self.reward_memory, indices=indices, updates=reward))

        # Add episode indices.
        with tf.control_dependencies(control_inputs=assignments):
            num_episodes = tf.count_nonzero(input_tensor=terminal, axis=0, dtype=util.tf_dtype('int'))
            assignment = tf.assign(
                ref=self.episode_indices[self.episode_count: self.episode_count + num_episodes],
                value=tf.boolean_mask(tensor=indices, mask=terminal)
            )

        # Increment episode count.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign_add(ref=self.episode_count, value=num_episodes)

        # Increment memory index.
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign(
                ref=self.episode_indices[-1],
                value=tf.where(self.memory_index + num_instances > self.capacity,
                               self.episode_indices[self.episode_count - 1], self.capacity - 1)
            )

        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = tf.assign(ref=self.memory_index, value=((self.memory_index + num_instances) % self.capacity))

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
        for name in sorted(self.states_memory):
            states[name] = tf.gather(params=self.states_memory[name], indices=indices)

        internals = dict()
        for name in sorted(self.internals_memory):
            internals[name] = tf.gather(params=self.internals_memory[name], indices=indices)

        actions = dict()
        for name in sorted(self.actions_memory):
            actions[name] = tf.gather(params=self.actions_memory[name], indices=indices)

        terminal = tf.gather(params=self.terminal_memory, indices=indices)
        reward = tf.gather(params=self.reward_memory, indices=indices)

        if self.include_next_states:
            assert util.rank(indices) == 1
            next_indices = (indices + 1) % self.capacity

            next_states = dict()
            for name in sorted(self.states_memory):
                next_states[name] = tf.gather(params=self.states_memory[name], indices=next_indices)

            next_internals = dict()
            for name in sorted(self.internals_memory):
                next_internals[name] = tf.gather(params=self.internals_memory[name], indices=next_indices)

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
