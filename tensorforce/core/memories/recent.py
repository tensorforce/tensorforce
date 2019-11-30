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

import tensorflow as tf

from tensorforce import util
from tensorforce.core.memories import Queue


class Recent(Queue):
    """
    Batching memory which always retrieves most recent experiences (specification key: `recent`).

    Args:
        name (string): Memory name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        capacity (int > 0): Memory capacity, in experience timesteps
            (<span style="color:#C00000"><b>required</b></span>).
        values_spec (specification): Values specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def tf_retrieve_timesteps(self, n, past_padding, future_padding):
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        # # Start index of oldest episode
        # oldest_episode_start = self.terminal_indices[0] + one + past_padding

        # # Number of timesteps (minus/plus one to prevent zero but allow capacity)
        # num_timesteps = self.buffer_index - oldest_episode_start - future_padding - one
        # num_timesteps = tf.math.mod(x=num_timesteps, y=capacity) + one

        # Check whether memory contains enough timesteps
        num_timesteps = tf.minimum(x=self.buffer_index, y=capacity) - past_padding - future_padding
        assertion = tf.debugging.assert_less_equal(x=n, y=num_timesteps)

        # Most recent timestep indices range
        with tf.control_dependencies(control_inputs=(assertion,)):  # Assertions in memory as warning!!!
            indices = tf.range(start=(self.buffer_index - n), limit=self.buffer_index)
            indices = tf.math.mod(x=(indices - future_padding), y=capacity)

        return indices

    def tf_retrieve_episodes(self, n):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        # Check whether memory contains enough episodes
        assertion = tf.debugging.assert_less_equal(x=n, y=self.episode_count)

        # Get start and limit index for most recent n episodes
        with tf.control_dependencies(control_inputs=(assertion,)):
            start = self.terminal_indices[self.episode_count - n]
            limit = self.terminal_indices[self.episode_count]
            # Increment terminal of previous episode
            start = start + one
            limit = limit + one

        # Correct limit index if smaller than start index
        limit = limit + tf.where(condition=(limit < start), x=capacity, y=zero)

        # Most recent episode indices range
        indices = tf.range(start=start, limit=limit)
        indices = tf.math.mod(x=indices, y=capacity)

        return indices

    # def tf_retrieve_sequences(self, n, sequence_length):
    #     one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
    #     capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

    #     # Start index of oldest episode
    #     oldest_episode_start = self.terminal_indices[0] + one

    #     # Number of sequences (minus/plus one to prevent zero but allow capacity-sequence_length)
    #     num_sequences = self.buffer_index - oldest_episode_start - sequence_length
    #     num_sequences = tf.math.mod(x=num_sequences, y=capacity) + one

    #     # Check whether memory contains enough sequences
    #     assertion = tf.debugging.assert_less_equal(x=n, y=num_sequences)

    #     # Most recent timestep indices range
    #     with tf.control_dependencies(control_inputs=(assertion,)):
    #         indices = tf.range(
    #             start=(self.buffer_index - n - sequence_length), limit=self.buffer_index
    #         )
    #         indices = tf.math.mod(x=indices, y=capacity)

    #     # ???????
    #     # sequence_indices = [tf.range(start=indices[n], limit=(indices[n] + sequence_length)) for k in range(n)]
    #     # sequence_indices = [indices[k: k + sequence_length] for k in tf.unstack(value=tf.range(start=0, limit=n), num=n)]
    #     sequence_indices = tf.expand_dims(input=tf.range(start=0, limit=n), axis=1)
    #     sequence_indices += tf.expand_dims(input=tf.range(start=0, limit=sequence_length), axis=0)
    #     sequence_indices = tf.reshape(tensor=sequence_indices, shape=(n * sequence_length,))
    #     # sequence_indices = tf.concat(values=sequence_indices, axis=0)  # tf.stack !!!!!
    #     terminal = tf.gather(params=self.buffers['terminal'], indices=indices)
    #     sequence_indices = tf.boolean_mask(
    #         tensor=sequence_indices, mask=tf.logical_not(x=terminal)
    #     )

    #     # Retrieve sequence indices
    #     sequences = self.retrieve(indices=sequence_indices)

    #     return sequence_indices, sequences
