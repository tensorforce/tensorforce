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


class Replay(Queue):
    """
    Memory which randomly retrieves experiences.
    """

    def tf_retrieve_timesteps(self, n):
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        # Start index of oldest episode
        oldest_episode_start = self.terminal_indices[0] + one

        # Number of timesteps (minus/plus one to prevent zero but allow capacity)
        num_timesteps = self.memory_index - oldest_episode_start - one
        num_timesteps = tf.mod(x=num_timesteps, y=capacity) + one

        # Check whether memory contains enough timesteps
        assertion = tf.debugging.assert_less_equal(x=n, y=num_timesteps)

        # Randomly sampled timestep indices
        with tf.control_dependencies(control_inputs=(assertion,)):
            indices = tf.random_uniform(
                shape=(n,), maxval=num_timesteps, dtype=util.tf_dtype(dtype='long')
            )
            indices = tf.mod(x=(self.memory_index - one - indices), y=capacity)

        # Retrieve timestep indices
        timesteps = self.retrieve_indices(indices=indices)

        return timesteps

        # if self.include_next_states:
        #     # Ensure consistent next state semantics for Q models.
        #     terminal = tf.gather(params=self.memories['terminal'], indices=indices)
        #     indices = tf.boolean_mask(tensor=indices, mask=tf.logical_not(x=terminal))

        #     # Simple rejection sampling in case masking out terminals yielded
        #     # no indices.
        #     def resample_fn():
        #         def cond(sampled_indices):
        #             # Any index contained after masking?
        #             return tf.reduce_any(input_tensor=(sampled_indices >= 0))

        #         def sampling_body(sampled_indices):
        #             # Resample. Note that we could also try up to fill
        #             # masked out indices.
        #             sampled_indices = tf.random_uniform(shape=(n,), maxval=num_timesteps, dtype=tf.int32)
        #             sampled_indices = (self.memory_index - 1 - sampled_indices) % self.capacity

        #             terminal = tf.gather(params=self.memories['terminal'], indices=sampled_indices)
        #             sampled_indices = tf.boolean_mask(tensor=sampled_indices, mask=tf.logical_not(x=terminal))

        #             return sampled_indices

        #         sampled_indices = self.while_loop(
        #             cond=cond,
        #             body=sampling_body,
        #             loop_vars=[indices],
        #             maximum_iterations=10
        #         )
        #         return sampled_indices

        #     # If there are still indices after masking, return these, otherwise resample.
        #     indices = self.cond(
        #         pred=tf.reduce_any(input_tensor=(indices >= 0)),
        #         true_fn=(lambda: indices),
        #         false_fn=resample_fn
        #     )
        #     return self.retrieve_indices(indices=indices)
        # else:
        #     # No masking necessary if next-state semantics are not relevant.
        #     return self.retrieve_indices(indices=indices)

    def tf_retrieve_episodes(self, n):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        # Check whether memory contains enough episodes
        assertion = tf.debugging.assert_less_equal(x=n, y=self.episode_count)

        # Get start and limit indices for randomly sampled n episodes
        with tf.control_dependencies(control_inputs=(assertion,)):
            random_terminal_indices = tf.random.uniform(
                shape=(n,), maxval=self.episode_count, dtype=util.tf_dtype(dtype='long')
            )
            starts = tf.gather(params=self.terminal_indices, indices=random_terminal_indices)
            limits = tf.gather(
                params=self.terminal_indices, indices=(random_terminal_indices + one)
            )
            # Increment terminal of previous episode
            starts = starts + one
            limits = limits + one

        # Correct limit indices if smaller than start indices
        zero_array = tf.fill(
            dims=(n,), value=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        )
        capacity_array = tf.fill(
            dims=(n,), value=tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))
        )
        limits = limits + tf.where(condition=(limits < starts), x=capacity_array, y=zero_array)

        # Concatenate randomly sampled episode indices ranges
        def cond(indices, i):
            return tf.math.greater_equal(x=i, y=n)

        def reduce_range_concat(indices, i):
            episode_indices = tf.range(start=starts[i], limit=limits[i])
            indices = tf.concat(values=(indices, episode_indices), axis=0)
            i = i + one
            return indices, i

        indices = tf.zeros(shape=(0,), dtype=util.tf_dtype(dtype='long'))
        indices, _ = self.while_loop(
            cond=cond, body=reduce_range_concat, loop_vars=(indices, zero),
            shape_invariants=(tf.TensorShape(dims=(None,)), zero.get_shape())
        )
        indices = tf.mod(x=indices, y=capacity)

        # Retrieve episode indices
        episodes = self.retrieve_indices(indices=indices)

        return episodes

    def tf_retrieve_sequences(self, n, sequence_length):
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        # Start index of oldest episode
        oldest_episode_start = self.terminal_indices[0] + one

        # Number of sequences (minus/plus one to prevent zero but allow capacity-sequence_length)
        num_sequences = self.memory_index - oldest_episode_start - sequence_length
        num_sequences = tf.mod(x=num_sequences, y=capacity) + one

        # Check whether memory contains enough sequences
        assertion = tf.debugging.assert_less_equal(x=n, y=num_sequences)

        # Randomly sampled timestep indices range
        with tf.control_dependencies(control_inputs=(assertion,)):
            indices = tf.random_uniform(
                shape=(n,), maxval=num_sequences, dtype=util.tf_dtype(dtype='long')
            )
            indices = tf.mod(x=(self.memory_index - 1 - indices - sequence_length), y=capacity)

        # ???????
        # sequence_indices = [tf.range(start=indices[n], limit=(indices[n] + sequence_length)) for k in range(n)]
        # sequence_indices = [indices[k: k + sequence_length] for k in tf.unstack(value=tf.range(start=0, limit=n), num=n)]
        sequence_indices = tf.expand_dims(input=tf.range(start=0, limit=n), axis=1)
        sequence_indices += tf.expand_dims(input=tf.range(start=0, limit=sequence_length), axis=0)
        sequence_indices = tf.reshape(tensor=sequence_indices, shape=(n * sequence_length,))
        # sequence_indices = tf.concat(values=sequence_indices, axis=0)  # tf.stack !!!!!
        terminal = tf.gather(params=self.memories['terminal'], indices=indices)
        sequence_indices = tf.boolean_mask(
            tensor=sequence_indices, mask=tf.logical_not(x=terminal)
        )
        return self.retrieve_indices(indices=sequence_indices)

        # Retrieve sequence indices
        sequences = self.retrieve_indices(indices=sequence_indices)

        return sequences
