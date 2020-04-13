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

from tensorforce.core import tf_function, tf_util
from tensorforce.core.memories import Queue


class Replay(Queue):
    """
    Replay memory which randomly retrieves experiences (specification key: `replay`).

    Args:
        capacity (int > 0): Memory capacity
            (<span style="color:#00C000"><b>default</b></span>: minimum capacity).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: CPU:0).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        values_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        min_capacity (int >= 0): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    @tf_function(num_args=3)
    def retrieve_timesteps(self, n, past_horizon, future_horizon):
        one = tf.constant(value=1, dtype='int')
        capacity = tf.constant(value=self.capacity, dtype='int')

        # Check whether memory contains at least one valid timestep
        num_timesteps = tf.math.minimum(x=self.buffer_index, y=capacity)
        num_timesteps -= past_horizon + future_horizon
        assertion = tf.debugging.assert_greater_equal(x=num_timesteps, y=one)

        # Randomly sampled timestep indices
        with tf.control_dependencies(control_inputs=(assertion,)):
            indices = tf.random.uniform(
                shape=(n,), maxval=num_timesteps, dtype=tf_util.get_dtype(type='int')
            )
            indices = tf.math.mod(
                x=(self.buffer_index - one - indices - future_horizon), y=capacity
            )

        return indices

    @tf_function(num_args=1)
    def retrieve_episodes(self, n):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')

        # Check whether memory contains at least one episode
        assertion = tf.debugging.assert_greater_equal(x=self.episode_count, y=one)

        # Get start and limit indices for randomly sampled n episodes
        with tf.control_dependencies(control_inputs=(assertion,)):
            random_terminal_indices = tf.random.uniform(
                shape=(n,), maxval=self.episode_count, dtype=tf_util.get_dtype(type='int')
            )
            starts = tf.gather(params=self.terminal_indices, indices=random_terminal_indices)
            limits = tf.gather(
                params=self.terminal_indices, indices=(random_terminal_indices + one)
            )
            # Increment terminal of previous episode
            starts = starts + one
            limits = limits + one

            # Correct limit indices if smaller than start indices
            zero_array = tf.fill(dims=(n,), value=tf_util.constant(value=0, dtype='int'))
            capacity_array = tf.fill(
                dims=(n,), value=tf_util.constant(value=self.capacity, dtype='int')
            )
            limits = limits + tf.where(condition=(limits < starts), x=capacity_array, y=zero_array)

            # Concatenate randomly sampled episode indices ranges
            def cond(indices, i):
                return tf.math.less(x=i, y=n)

            def reduce_range_concat(indices, i):
                episode_indices = tf.range(start=starts[i], limit=limits[i])
                indices = tf.concat(values=(indices, episode_indices), axis=0)
                i = i + one
                return indices, i

            indices = tf_util.zeros(shape=(0,), dtype='int')
            indices, _ = self.while_loop(
                cond=cond, body=reduce_range_concat, loop_vars=(indices, zero),
                shape_invariants=(tf.TensorShape(dims=(None,)), zero.get_shape()), back_prop=False
            )
            indices = tf.math.mod(x=indices, y=capacity)

        return indices
