# Copyright 2020 Tensorforce Team. All Rights Reserved.
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
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        values_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        min_capacity (int >= 0): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    @tf_function(num_args=3)
    def retrieve_timesteps(self, *, n, past_horizon, future_horizon):
        one = tf_util.constant(value=1, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')

        # Check whether memory contains at least one valid timestep
        num_timesteps = tf.math.minimum(x=self.buffer_index, y=capacity)
        num_timesteps -= (past_horizon + future_horizon)
        num_timesteps = tf.math.maximum(x=num_timesteps, y=self.episode_count)

        # Check whether memory contains at least one timestep
        assertions = list()
        if self.config.create_tf_assertions:
            assertions.append(tf.debugging.assert_greater_equal(x=num_timesteps, y=one))

        # Randomly sampled timestep indices
        with tf.control_dependencies(control_inputs=assertions):
            n = tf.math.minimum(x=n, y=num_timesteps)
            indices = tf.random.uniform(
                shape=(n,), maxval=num_timesteps, dtype=tf_util.get_dtype(type='int')
            )
            indices = tf.math.mod(
                x=(self.buffer_index - one - indices - future_horizon), y=capacity
            )

        return indices

    @tf_function(num_args=1)
    def retrieve_episodes(self, *, n):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')

        # Check whether memory contains at least one episode
        assertions = list()
        if self.config.create_tf_assertions:
            assertions.append(tf.debugging.assert_greater_equal(x=self.episode_count, y=one))

        # Get start and limit indices for randomly sampled n episodes
        with tf.control_dependencies(control_inputs=assertions):
            n = tf.math.minimum(x=n, y=self.episode_count)
            random_indices = tf.random.uniform(
                shape=(n,), maxval=self.episode_count, dtype=tf_util.get_dtype(type='int')
            )

            # (Increment terminal of previous episode)
            starts = tf.gather(params=self.terminal_indices, indices=random_indices) + one
            limits = tf.gather(params=self.terminal_indices, indices=(random_indices + one)) + one

            # Correct limit index if smaller than start index
            limits = limits + tf.where(condition=(limits < starts), x=capacity, y=zero)

            # Random episode indices ranges
            indices = tf.ragged.range(starts=starts, limits=limits).values
            indices = tf.math.mod(x=indices, y=capacity)

        return indices
