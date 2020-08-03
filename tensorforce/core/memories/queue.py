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

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import TensorDict, TensorSpec, tf_function, tf_util, VariableDict
from tensorforce.core.memories import Memory


class Queue(Memory):
    """
    Base class for memories organized as a queue / circular buffer.

    Args:
        capacity (int > 0): Memory capacity
            (<span style="color:#00C000"><b>default</b></span>: minimum capacity).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: CPU:0).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        values_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        min_capacity (int >= 0): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    # (requires capacity as first argument)
    def __init__(
        self, capacity=None, *, device='CPU:0', name=None, values_spec=None, min_capacity=None
    ):
        super().__init__(
            device=device, name=name, values_spec=values_spec, min_capacity=min_capacity
        )

        if capacity is None:
            if self.min_capacity is None:
                raise TensorforceError.required(
                    name='memory', argument='capacity', condition='unknown minimum capacity'
                )
            else:
                self.capacity = self.min_capacity
        elif capacity < self.min_capacity:
            raise TensorforceError.value(
                name='memory', argument='capacity', value=capacity,
                hint=('< minimum capacity ' + str(self.min_capacity))
            )
        else:
            self.capacity = capacity

    def initialize(self):
        super().initialize()

        # Value buffers
        def function(name, spec):
            spec = TensorSpec(type=spec.type, shape=((self.capacity,) + spec.shape))
            if name == 'terminal':
                initializer = np.zeros(shape=(self.capacity,), dtype=spec.np_type())
                initializer[-1] = 1
            else:
                initializer = 'zeros'
            return self.variable(
                name=(name.replace('/', '_') + '-buffer'), spec=spec, initializer=initializer,
                is_trainable=False, is_saved=True
            )

        self.buffers = self.values_spec.fmap(function=function, cls=VariableDict, with_names=True)

        # Buffer index (modulo capacity, next index to write to)
        self.buffer_index = self.variable(
            name='buffer-index', spec=TensorSpec(type='int'), initializer='zeros',
            is_trainable=False, is_saved=True
        )

        # Terminal indices
        # (oldest episode terminals first, initially the only terminal is last index)
        initializer = np.zeros(shape=(self.capacity + 1,), dtype=util.np_dtype(dtype='int'))
        initializer[0] = self.capacity - 1
        self.terminal_indices = self.variable(
            name='terminal-indices', spec=TensorSpec(type='int', shape=(self.capacity + 1,)),
            initializer=initializer, is_trainable=False, is_saved=True
        )

        # Episode count
        self.episode_count = self.variable(
            name='episode-count', spec=TensorSpec(type='int'), initializer='zeros',
            is_trainable=False, is_saved=True
        )

    @tf_function(num_args=0)
    def reset(self):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        three = tf_util.constant(value=3, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')
        last_index = tf.math.mod(x=(self.buffer_index - one), y=capacity)

        def correct_terminal():
            # Replace last observation terminal marker with abort terminal
            dependencies = list()
            two = tf_util.constant(value=2, dtype='int')
            sparse_delta = tf.IndexedSlices(values=two, indices=last_index)
            dependencies.append(self.buffers['terminal'].scatter_update(sparse_delta=sparse_delta))
            sparse_delta = tf.IndexedSlices(values=last_index, indices=(self.episode_count + one))
            dependencies.append(self.terminal_indices.scatter_update(sparse_delta=sparse_delta))
            with tf.control_dependencies(control_inputs=dependencies):
                return self.episode_count.assign_add(delta=one, read_value=False)

        last_terminal = tf.gather(params=self.buffers['terminal'], indices=last_index)
        is_incorrect = tf.math.equal(x=last_terminal, y=three)
        corrected = tf.cond(pred=is_incorrect, true_fn=correct_terminal, false_fn=tf.no_op)

        with tf.control_dependencies(control_inputs=(corrected,)):
            assertions = [corrected]
            if self.config.create_tf_assertions:
                # general check: all terminal indices true
                assertions.append(tf.debugging.assert_equal(
                    x=tf.reduce_all(
                        input_tensor=tf.gather(
                            params=tf.math.greater(x=self.buffers['terminal'], y=zero),
                            indices=self.terminal_indices[:self.episode_count + one]
                        )
                    ),
                    y=tf_util.constant(value=True, dtype='bool'),
                    message="Memory consistency check."
                ))
                # general check: only terminal indices true
                assertions.append(tf.debugging.assert_equal(
                    x=tf.math.count_nonzero(
                        input=self.buffers['terminal'], dtype=tf_util.get_dtype(type='int')
                    ),
                    y=(self.episode_count + one), message="Memory consistency check."
                ))

        with tf.control_dependencies(control_inputs=assertions):
            return one < zero

    @tf_function(num_args=6)
    def enqueue(self, *, states, internals, auxiliaries, actions, terminal, reward):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        three = tf_util.constant(value=3, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')
        num_timesteps = tf_util.cast(x=tf.shape(input=terminal)[0], dtype='int')

        last_index = tf.math.mod(x=(self.buffer_index - one), y=capacity)

        def correct_terminal():
            # Remove last observation terminal marker
            sparse_delta = tf.IndexedSlices(values=zero, indices=last_index)
            assignment = self.buffers['terminal'].scatter_update(sparse_delta=sparse_delta)
            with tf.control_dependencies(control_inputs=(assignment,)):
                return last_index < zero

        last_terminal = tf.gather(params=self.buffers['terminal'], indices=last_index)
        is_incorrect = tf.math.equal(x=last_terminal, y=three)
        corrected = tf.cond(pred=is_incorrect, true_fn=correct_terminal, false_fn=tf.no_op)

        # Assertions
        last_terminal = tf.concat(values=([zero], terminal), axis=0)[-1]
        assertions = [corrected]
        if self.config.create_tf_assertions:
            with tf.control_dependencies(control_inputs=(corrected,)):
                # check: number of timesteps fit into effectively available buffer
                assertions.append(tf.debugging.assert_less_equal(
                    x=num_timesteps, y=capacity, message="Memory does not have enough capacity."
                ))
                # at most one terminal
                assertions.append(tf.debugging.assert_less_equal(
                    x=tf.math.count_nonzero(input=terminal, dtype=tf_util.get_dtype(type='int')),
                    y=one, message="Timesteps contain more than one terminal."
                ))
                # if terminal, last timestep in batch
                assertions.append(tf.debugging.assert_equal(
                    x=tf.math.reduce_any(input_tensor=tf.math.greater(x=terminal, y=zero)),
                    y=tf.math.greater(x=last_terminal, y=zero),
                    message="Terminal is not the last timestep."
                ))
                # general check: all terminal indices true
                assertions.append(tf.debugging.assert_equal(
                    x=tf.reduce_all(
                        input_tensor=tf.gather(
                            params=tf.math.greater(x=self.buffers['terminal'], y=zero),
                            indices=self.terminal_indices[:self.episode_count + one]
                        )
                    ),
                    y=tf_util.constant(value=True, dtype='bool'),
                    message="Memory consistency check."
                ))
                # general check: only terminal indices true
                assertions.append(tf.debugging.assert_equal(
                    x=tf.math.count_nonzero(
                        input=self.buffers['terminal'], dtype=tf_util.get_dtype(type='int')
                    ),
                    y=(self.episode_count + one), message="Memory consistency check."
                ))

        # Buffer indices to overwrite
        with tf.control_dependencies(control_inputs=assertions):
            overwritten_indices = tf.range(
                start=self.buffer_index, limit=(self.buffer_index + num_timesteps)
            )
            overwritten_indices = tf.math.mod(x=overwritten_indices, y=capacity)

            # Count number of overwritten episodes
            num_episodes = tf.math.count_nonzero(
                input=tf.gather(params=self.buffers['terminal'], indices=overwritten_indices),
                axis=0, dtype=tf_util.get_dtype(type='int')
            )

            # Shift remaining terminal indices accordingly
            index = self.episode_count + one
            assertions = list()
            if self.config.create_tf_assertions:
                assertions.append(tf.debugging.assert_greater_equal(
                    x=index, y=num_episodes, message="Memory episode overwriting check."
                ))

        with tf.control_dependencies(control_inputs=assertions):
            sparse_delta = tf.IndexedSlices(
                values=self.terminal_indices[num_episodes: index],
                indices=tf.range(index - num_episodes)
            )
            assignment = self.terminal_indices.scatter_update(sparse_delta=sparse_delta)

        # Decrement episode count accordingly
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = self.episode_count.assign_sub(delta=num_episodes, read_value=False)

        # Write new observations
        with tf.control_dependencies(control_inputs=(assignment,)):
            # Add last observation terminal marker
            corrected_terminal = tf.where(
                condition=tf.math.equal(x=terminal[-1:], y=zero), x=three, y=terminal[-1:]
            )
            corrected_terminal = tf.concat(values=(terminal[:-1], corrected_terminal), axis=0)
            values = TensorDict(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=corrected_terminal, reward=reward
            )
            indices = tf.range(start=self.buffer_index, limit=(self.buffer_index + num_timesteps))
            indices = tf.math.mod(x=indices, y=capacity)

            def function(buffer, value):
                sparse_delta = tf.IndexedSlices(values=value, indices=indices)
                return buffer.scatter_update(sparse_delta=sparse_delta)

            assignments = self.buffers.fmap(function=function, cls=list, zip_values=values)

        # Increment buffer index
        with tf.control_dependencies(control_inputs=assignments):
            assignment = self.buffer_index.assign_add(delta=num_timesteps, read_value=False)

        # Count number of new episodes
        with tf.control_dependencies(control_inputs=(assignment,)):
            num_new_episodes = tf.math.count_nonzero(
                input=terminal, dtype=tf_util.get_dtype(type='int')
            )

            # Write new terminal indices
            new_terminal_indices = tf.boolean_mask(
                tensor=overwritten_indices, mask=tf.math.greater(x=terminal, y=zero)
            )
            start = self.episode_count + one
            sparse_delta = tf.IndexedSlices(
                values=new_terminal_indices,
                indices=tf.range(start=start, limit=(start + num_new_episodes))
            )
            assignment = self.terminal_indices.scatter_update(sparse_delta=sparse_delta)

        # Increment episode count accordingly
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = self.episode_count.assign_add(delta=num_new_episodes)
            return assignment < zero

    @tf_function(num_args=1)
    def retrieve(self, *, indices, values):
        assert isinstance(values, tuple)
        function = (lambda x: tf.gather(params=x, indices=indices))
        return self.buffers[values].fmap(function=function, cls=TensorDict)

    @tf_function(num_args=2)
    def predecessors(self, *, indices, horizon, sequence_values, initial_values):
        assert isinstance(sequence_values, tuple)
        assert isinstance(initial_values, tuple)

        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')

        def body(lengths, predecessor_indices, mask):
            previous_index = tf.math.mod(x=(predecessor_indices[:, :1] - one), y=capacity)
            predecessor_indices = tf.concat(values=(previous_index, predecessor_indices), axis=1)
            previous_terminal = tf.gather(params=self.buffers['terminal'], indices=previous_index)
            is_not_terminal = tf.math.logical_and(
                x=tf.math.logical_not(x=tf.math.greater(x=previous_terminal, y=zero)),
                y=mask[:, :1]
            )
            mask = tf.concat(values=(is_not_terminal, mask), axis=1)
            is_not_terminal = tf.squeeze(input=is_not_terminal, axis=1)
            zeros = tf.zeros_like(input=is_not_terminal, dtype=tf_util.get_dtype(type='int'))
            ones = tf.ones_like(input=is_not_terminal, dtype=tf_util.get_dtype(type='int'))
            lengths += tf.where(condition=is_not_terminal, x=ones, y=zeros)
            return lengths, predecessor_indices, mask

        lengths = tf.ones_like(input=indices, dtype=tf_util.get_dtype(type='int'))
        predecessor_indices = tf.expand_dims(input=indices, axis=1)
        mask = tf.ones_like(input=predecessor_indices, dtype=tf_util.get_dtype(type='bool'))
        shape = tf.TensorShape(dims=((None, None)))

        lengths, predecessor_indices, mask = tf.while_loop(
            cond=tf_util.always_true, body=body, loop_vars=(lengths, predecessor_indices, mask),
            shape_invariants=(lengths.get_shape(), shape, shape),
            maximum_iterations=tf_util.int32(x=horizon)
        )

        predecessor_indices = tf.reshape(tensor=predecessor_indices, shape=(-1,))
        mask = tf.reshape(tensor=mask, shape=(-1,))
        predecessor_indices = tf.boolean_mask(tensor=predecessor_indices, mask=mask, axis=0)

        assertions = list()
        if self.config.create_tf_assertions:
            assertions.append(tf.debugging.assert_greater_equal(
                x=tf.math.mod(x=(predecessor_indices - self.buffer_index), y=capacity), y=zero,
                message="Predecessor check."
            ))

        with tf.control_dependencies(control_inputs=assertions):
            function = (lambda buffer: tf.gather(params=buffer, indices=predecessor_indices))
            sequence_values = self.buffers[sequence_values].fmap(function=function, cls=TensorDict)

            starts = tf.math.cumsum(x=lengths, exclusive=True)
            initial_indices = tf.gather(params=predecessor_indices, indices=starts)
            function = (lambda buffer: tf.gather(params=buffer, indices=initial_indices))
            initial_values = self.buffers[initial_values].fmap(function=function, cls=TensorDict)

        if len(sequence_values) == 0:
            if len(initial_values) == 0:
                return lengths
            else:
                return lengths, initial_values

        elif len(initial_values) == 0:
            return tf.stack(values=(starts, lengths), axis=1), sequence_values

        else:
            return tf.stack(values=(starts, lengths), axis=1), sequence_values, initial_values

    @tf_function(num_args=2)
    def successors(self, *, indices, horizon, sequence_values, final_values):
        assert isinstance(sequence_values, tuple)
        assert isinstance(final_values, tuple)

        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')
        capacity = tf_util.constant(value=self.capacity, dtype='int')

        def body(lengths, successor_indices, mask):
            current_index = successor_indices[:, -1:]
            current_terminal = tf.gather(params=self.buffers['terminal'], indices=current_index)
            is_not_terminal = tf.math.logical_and(
                x=tf.math.logical_not(x=tf.math.greater(x=current_terminal, y=zero)),
                y=mask[:, -1:]
            )
            next_index = tf.math.mod(x=(current_index + one), y=capacity)
            successor_indices = tf.concat(values=(successor_indices, next_index), axis=1)
            mask = tf.concat(values=(mask, is_not_terminal), axis=1)
            is_not_terminal = tf.squeeze(input=is_not_terminal, axis=1)
            zeros = tf.zeros_like(input=is_not_terminal, dtype=tf_util.get_dtype(type='int'))
            ones = tf.ones_like(input=is_not_terminal, dtype=tf_util.get_dtype(type='int'))
            lengths += tf.where(condition=is_not_terminal, x=ones, y=zeros)
            return lengths, successor_indices, mask

        lengths = tf.ones_like(input=indices, dtype=tf_util.get_dtype(type='int'))
        successor_indices = tf.expand_dims(input=indices, axis=1)
        mask = tf.ones_like(input=successor_indices, dtype=tf_util.get_dtype(type='bool'))
        shape = tf.TensorShape(dims=((None, None)))

        lengths, successor_indices, mask = tf.while_loop(
            cond=tf_util.always_true, body=body, loop_vars=(lengths, successor_indices, mask),
            shape_invariants=(lengths.get_shape(), shape, shape),
            maximum_iterations=tf_util.int32(x=horizon)
        )

        successor_indices = tf.reshape(tensor=successor_indices, shape=(-1,))
        mask = tf.reshape(tensor=mask, shape=(-1,))
        successor_indices = tf.boolean_mask(tensor=successor_indices, mask=mask, axis=0)

        assertions = list()
        if self.config.create_tf_assertions:
            assertions.append(tf.debugging.assert_greater_equal(
                x=tf.math.mod(x=(self.buffer_index - one - successor_indices), y=capacity), y=zero,
                message="Successor check."
            ))

        with tf.control_dependencies(control_inputs=assertions):
            function = (lambda buffer: tf.gather(params=buffer, indices=successor_indices))
            sequence_values = self.buffers[sequence_values].fmap(function=function, cls=TensorDict)

            starts = tf.math.cumsum(x=lengths, exclusive=True)
            ends = tf.math.cumsum(x=lengths) - one
            final_indices = tf.gather(params=successor_indices, indices=ends)
            function = (lambda buffer: tf.gather(params=buffer, indices=final_indices))
            final_values = self.buffers[final_values].fmap(function=function, cls=TensorDict)

        if len(sequence_values) == 0:
            if len(final_values) == 0:
                return lengths
            else:
                return lengths, final_values

        elif len(final_values) == 0:
            return tf.stack(values=(starts, lengths), axis=1), sequence_values

        else:
            return tf.stack(values=(starts, lengths), axis=1), sequence_values, final_values
