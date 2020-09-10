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


class Recent(Queue):
    """
    Batching memory which always retrieves most recent experiences (specification key: `recent`).

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

        # Most recent timestep indices range
        with tf.control_dependencies(control_inputs=assertions):
            n = tf.math.minimum(x=n, y=num_timesteps)
            indices = tf.range(start=(self.buffer_index - n), limit=self.buffer_index)
            indices = tf.math.mod(x=(indices - future_horizon), y=capacity)

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

        # Get start and limit index for most recent n episodes
        with tf.control_dependencies(control_inputs=assertions):
            n = tf.math.minimum(x=n, y=self.episode_count)

            # (Increment terminal of previous episode)
            start = self.terminal_indices[self.episode_count - n] + one
            limit = self.terminal_indices[self.episode_count] + one

            # Correct limit index if smaller than start index
            limit = limit + tf.where(condition=(limit < start), x=capacity, y=zero)

            # Most recent episode indices range
            indices = tf.range(start=start, limit=limit)
            indices = tf.math.mod(x=indices, y=capacity)

        return indices
