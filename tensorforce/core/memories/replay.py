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
            states (dict): States specification.
            internals (dict): Internal states specification.
            actions (dict): Actions specification.
            include_next_states (bool): Include subsequent state if true.
            capacity (int): Memory capacity (number of state/internals/action/(next-state)? records).
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
        num_timesteps = (self.memory_index - self.episode_indices[-1] - 2) % self.capacity + 1
        indices = tf.random_uniform(shape=(n,), maxval=num_timesteps, dtype=tf.int32)
        indices = (self.memory_index - 1 - indices) % self.capacity

        if self.include_next_states:
            # Ensure consistent next state semantics for Q models.
            terminal = tf.gather(params=self.terminal_memory, indices=indices)
            indices = tf.boolean_mask(tensor=indices, mask=tf.logical_not(x=terminal))

            # Simple rejection sampling in case masking out terminals yielded
            # no indices.
            def resample_fn():
                def cond(sampled_indices):
                    # Any index contained after masking?
                    return tf.reduce_any(input_tensor=(sampled_indices >= 0))

                def sampling_body(sampled_indices):
                    # Resample. Note that we could also try up to fill
                    # masked out indices.
                    sampled_indices = tf.random_uniform(shape=(n,), maxval=num_timesteps, dtype=tf.int32)
                    sampled_indices = (self.memory_index - 1 - sampled_indices) % self.capacity

                    terminal = tf.gather(params=self.terminal_memory, indices=sampled_indices)
                    sampled_indices = tf.boolean_mask(tensor=sampled_indices, mask=tf.logical_not(x=terminal))

                    return sampled_indices

                sampled_indices = tf.while_loop(
                    cond=cond,
                    body=sampling_body,
                    loop_vars=[indices],
                    maximum_iterations=10
                )
                return sampled_indices

            # If there are still indices after masking, return these, otherwise resample.
            indices = tf.cond(
                pred=tf.reduce_any(input_tensor=(indices >= 0)),
                true_fn=(lambda: indices),
                false_fn=resample_fn
            )
            return self.retrieve_indices(indices=indices)
        else:
            # No masking necessary if next-state semantics are not relevant.
            return self.retrieve_indices(indices=indices)

    def tf_retrieve_episodes(self, n):
        asserts = [
            tf.assert_greater(x=self.episode_count, y=0, message="nothing stored yet")
        ]
        with tf.control_dependencies(control_inputs=asserts):
            # TODO: Should we use tf.random_shuffle(tf.range(count))[:n] for better sampling?
            random_episode_indices = tf.random_uniform(shape=(n,), maxval=self.episode_count, dtype=tf.int32)
        # -1 should translate to capacity, the episode_indices has length capacity + 1
        starts = tf.gather(params=self.episode_indices, indices=(random_episode_indices - 1) % (self.capacity + 1)) + 1
        limits = tf.gather(params=self.episode_indices, indices=random_episode_indices) + 1
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
