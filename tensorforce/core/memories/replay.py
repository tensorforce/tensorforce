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

from tensorforce.core.memories import Queue


class Replay(Queue):
    """
    Memory which randomly retrieves experiences.
    """

    def __init__(self, states, internals, actions, include_next_states, capacity, scope='replay', summary_labels=None):
        """
        Replay memory.

        Args:
            states: States specification.
            internals: Internal states specification.
            actions: Actions specification.
            include_next_states: Include subsequent state if true.
            capacity: Memory capacity.
        """
        super(Replay, self).__init__(
            states=states,
            internals=internals,
            actions=actions,
            include_next_states=include_next_states,
            capacity=capacity,
            scope=scope,
            summary_labels=summary_labels
        )

    def tf_retrieve_timesteps(self, n):
        num_timesteps = (self.memory_index - self.episode_indices[0] - 2) % self.capacity + 1
        indices = tf.random_uniform(shape=(n,), maxval=num_timesteps, dtype=tf.int32)
        indices = (self.memory_index - 1 - indices) % self.capacity
        terminal = tf.gather(params=self.terminal_memory, indices=indices)
        indices = tf.boolean_mask(tensor=indices, mask=tf.logical_not(x=terminal))
        return self.retrieve_indices(indices=indices)

    def tf_retrieve_episodes(self, n):
        random_episode_indices = tf.random_uniform(shape=(n,), maxval=(self.episode_count + 1), dtype=tf.int32)
        starts = tf.gather(params=self.episode_indices, indices=random_episode_indices) + 1
        limits = tf.gather(params=self.episode_indices, indices=(random_episode_indices + 1))
        limits += tf.where(
            condition=(starts < limits),
            x=tf.constant(value=0, shape=(n,)),
            y=tf.constant(value=self.capacity, shape=(n,))
        )
        episodes = [tf.range(start=starts[k], limit=limits[k]) for k in range(n)]
        indices = tf.concat(values=episodes, axis=0) % self.capacity
        return self.retrieve_indices(indices=indices)

    def tf_retrieve_sequences(self, n, sequence_length):
        num_sequences = (self.memory_index - self.episode_indices[0] - 2 - sequence_length + 1) % self.capacity + 1
        indices = tf.random_uniform(shape=(n,), maxval=num_sequences, dtype=tf.int32)
        indices = (self.memory_index - 1 - indices - sequence_length) % self.capacity
        sequence_indices = [tf.range(start=indices[k], limit=(indices[k] + sequence_length)) for k in range(n)]
        sequence_indices = tf.concat(values=sequence_indices, axis=0) % self.capacity  # tf.stack !!!!!
        terminal = tf.gather(params=self.terminal_memory, indices=indices)
        sequence_indices = tf.boolean_mask(tensor=sequence_indices, mask=tf.logical_not(x=terminal))
        return self.retrieve_indices(indices=sequence_indices)
