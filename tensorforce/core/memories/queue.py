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

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core.memories import Memory
from tensorforce.core.utils import CircularBuffer


class Queue(CircularBuffer, Memory):
    """
    Base class for memories organized as a queue / circular buffer.

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

    def __init__(self, name, capacity, values_spec, device=None, summary_labels=None):
        # Terminal initialization has to agree with terminal_indices
        terminal_initializer = np.zeros(shape=(capacity,), dtype=util.np_dtype(dtype='long'))
        terminal_initializer[-1] = 1
        initializers = OrderedDict(terminal=terminal_initializer)

        super().__init__(
            name=name, capacity=capacity, values_spec=values_spec, initializers=initializers,
            device=device, summary_labels=summary_labels
        )

    def tf_initialize(self):
        super().tf_initialize()

        # Terminal indices
        # (oldest episode terminals first, initially the only terminal is last index)
        initializer = np.zeros(shape=(self.capacity + 1,), dtype=util.np_dtype(dtype='long'))
        initializer[0] = self.capacity - 1
        self.terminal_indices = self.add_variable(
            name='terminal-indices', dtype='long', shape=(self.capacity + 1,), is_trainable=False,
            initializer=initializer
        )

        # Episode count
        self.episode_count = self.add_variable(
            name='episode-count', dtype='long', shape=(), is_trainable=False, initializer='zeros'
        )

    def tf_enqueue(self, states, internals, auxiliaries, actions, terminal, reward):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))
        num_timesteps = tf.shape(input=terminal, out_type=util.tf_dtype(dtype='long'))[0]

        # Max capacity
        latest_terminal_index = self.terminal_indices[self.episode_count]
        max_capacity = self.buffer_index - latest_terminal_index - one
        max_capacity = capacity - (tf.math.mod(x=max_capacity, y=capacity) + one)

        # Assertions
        assertions = [
            # check: number of timesteps fit into effectively available buffer
            tf.debugging.assert_less_equal(x=num_timesteps, y=max_capacity),
            # at most one terminal
            tf.debugging.assert_less_equal(
                x=tf.math.count_nonzero(input=terminal, dtype=util.tf_dtype(dtype='long')), y=one
            ),
            # if terminal, last timestep in batch
            tf.debugging.assert_equal(
                x=tf.math.reduce_any(input_tensor=tf.math.greater(x=terminal, y=zero)),
                y=tf.math.greater(x=terminal[-1], y=zero)
            ),
            # general check: all terminal indices true
            tf.debugging.assert_equal(
                x=tf.reduce_all(
                    input_tensor=tf.gather(
                        params=tf.math.greater(x=self.buffers['terminal'], y=zero),
                        indices=self.terminal_indices[:self.episode_count + one]
                    )
                ),
                y=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
            ),
            # general check: only terminal indices true
            tf.debugging.assert_equal(
                x=tf.math.count_nonzero(
                    input=self.buffers['terminal'], dtype=util.tf_dtype(dtype='long')
                ),
                y=(self.episode_count + one)
            ),
            # # general check: no terminal after last
            # tf.debugging.assert_equal(
            #     x=tf.reduce_any(
            #         input_tensor=tf.gather(
            #             params=self.buffers['terminal'], indices=tf.mathmod(
            #                 x=(self.terminal_indices[self.episode_count] + one + tf.range(
            #                     start=tf.math.mod(
            #                         x=(self.buffer_index - self.terminal_indices[self.episode_count] - one),
            #                         y=capacity
            #                     )
            #                 )),
            #                 y=capacity
            #             )
            #         )
            #     ),
            #     y=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))
            # )
        ]

        # Buffer indices to overwrite
        with tf.control_dependencies(control_inputs=assertions):
            indices = tf.range(start=self.buffer_index, limit=(self.buffer_index + num_timesteps))
            indices = tf.math.mod(x=indices, y=capacity)

            # Count number of overwritten episodes
            num_episodes = tf.math.count_nonzero(
                input=tf.gather(params=self.buffers['terminal'], indices=indices), axis=0,
                dtype=util.tf_dtype(dtype='long')
            )

        # Shift remaining terminal indices accordingly
        limit_index = self.episode_count + one
        assertion = tf.debugging.assert_greater_equal(x=limit_index, y=num_episodes)

        with tf.control_dependencies(control_inputs=(assertion,)):
            assignment = tf.compat.v1.assign(
                ref=self.terminal_indices[:limit_index - num_episodes],
                value=self.terminal_indices[num_episodes: limit_index]
            )

        # Decrement episode count accordingly
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = self.episode_count.assign_sub(delta=num_episodes, read_value=False)

        # # Write new observations
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = super().tf_enqueue(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )

        # # Write new observations
        # with tf.control_dependencies(control_inputs=(assignment,)):
        #     assignments = list()
        #     for name, state in states.items():
        #         assignments.append(
        #             tf.scatter_update(ref=self.memories[name], indices=indices, updates=state)
        #         )
        #     for name, internal in internals.items():
        #         assignments.append(
        #             tf.scatter_update(ref=self.memories[name], indices=indices, updates=internal)
        #         )
        #     for name, action in actions.items():
        #         assignments.append(
        #             tf.scatter_update(ref=self.memories[name], indices=indices, updates=action)
        #         )
        #     assignments.append(
        #         tf.scatter_update(ref=self.memories['terminal'], indices=indices, updates=terminal)
        #     )
        #     assignments.append(
        #         tf.scatter_update(ref=self.memories['reward'], indices=indices, updates=reward)
        #     )

        # # Increment memory index
        # with tf.control_dependencies(control_inputs=assignments):
        #     new_memory_index = tf.math.mod(x=(self.memory_index + num_timesteps), y=capacity)
        #     assignment = self.memory_index.assign(value=new_memory_index)

        # Count number of new episodes
        with tf.control_dependencies(control_inputs=(assignment,)):
            num_new_episodes = tf.math.count_nonzero(
                input=terminal, dtype=util.tf_dtype(dtype='long')
            )

            # Write new terminal indices
            limit_index = self.episode_count + one
            assignment = tf.compat.v1.assign(
                ref=self.terminal_indices[limit_index: limit_index + num_new_episodes],
                value=tf.boolean_mask(tensor=indices, mask=tf.math.greater(x=terminal, y=zero))
            )

        # Increment episode count accordingly
        with tf.control_dependencies(control_inputs=(assignment,)):
            assignment = self.episode_count.assign_add(delta=num_new_episodes, read_value=False)

        with tf.control_dependencies(control_inputs=(assignment,)):
            return util.no_operation()

    def tf_retrieve(self, indices, values):
        # if values is None:
        #     is_single_value = False
        #     values = ('states', 'internals', 'actions', 'terminal', 'reward')
        if isinstance(values, str):
            is_single_value = True
            values = [values]
        else:
            is_single_value = False
            values = list(values)

        # Retrieve values
        for n, name in enumerate(values):
            if util.is_nested(name=name):
                value = OrderedDict()
                for inner_name in self.values_spec[name]:
                    value[inner_name] = tf.gather(
                        params=self.buffers[name][inner_name], indices=indices
                    )
            else:
                value = tf.gather(params=self.buffers[name], indices=indices)
            values[n] = value

        # # Stop gradients
        # values = util.fmap(function=tf.stop_gradient, xs=values)

        # Return values or single value
        if is_single_value:
            return values[0]
        else:
            return values

    def tf_predecessors(self, indices, horizon, sequence_values=(), initial_values=()):
        if sequence_values == () and initial_values == ():
            raise TensorforceError.unexpected()

        if isinstance(sequence_values, str):
            is_single_sequence_value = True
            sequence_values = [sequence_values]
        else:
            is_single_sequence_value = False
            sequence_values = list(sequence_values)
        if isinstance(initial_values, str):
            is_single_initial_value = True
            initial_values = [initial_values]
        else:
            is_single_initial_value = False
            initial_values = list(initial_values)

        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        def body(lengths, predecessor_indices, mask):
            previous_index = tf.math.mod(x=(predecessor_indices[:, :1] - one), y=capacity)
            predecessor_indices = tf.concat(values=(previous_index, predecessor_indices), axis=1)
            previous_terminal = self.retrieve(indices=previous_index, values='terminal')
            is_not_terminal = tf.math.logical_and(
                x=tf.math.logical_not(x=tf.math.greater(x=previous_terminal, y=zero)),
                y=mask[:, :1]
            )
            mask = tf.concat(values=(is_not_terminal, mask), axis=1)
            is_not_terminal = tf.squeeze(input=is_not_terminal, axis=1)
            zeros = tf.zeros_like(input=is_not_terminal, dtype=util.tf_dtype(dtype='long'))
            ones = tf.ones_like(input=is_not_terminal, dtype=util.tf_dtype(dtype='long'))
            lengths += tf.where(condition=is_not_terminal, x=ones, y=zeros)
            return lengths, predecessor_indices, mask

        lengths = tf.ones_like(input=indices, dtype=util.tf_dtype(dtype='long'))
        predecessor_indices = tf.expand_dims(input=indices, axis=1)
        mask = tf.ones_like(input=predecessor_indices, dtype=util.tf_dtype(dtype='bool'))
        shape = tf.TensorShape(dims=((None, None)))

        lengths, predecessor_indices, mask = self.while_loop(
            cond=util.tf_always_true, body=body,
            loop_vars=(lengths, predecessor_indices, mask),
            shape_invariants=(lengths.get_shape(), shape, shape), back_prop=False,
            maximum_iterations=horizon
        )

        predecessor_indices = tf.reshape(tensor=predecessor_indices, shape=(-1,))
        mask = tf.reshape(tensor=mask, shape=(-1,))
        predecessor_indices = tf.boolean_mask(tensor=predecessor_indices, mask=mask, axis=0)

        assertion = tf.debugging.assert_greater_equal(
            x=tf.math.mod(x=(predecessor_indices - self.buffer_index), y=capacity), y=zero
        )

        with tf.control_dependencies(control_inputs=(assertion,)):
            starts = tf.math.cumsum(x=lengths, exclusive=True)
            initial_indices = tf.gather(params=predecessor_indices, indices=starts)

            for n, name in enumerate(sequence_values):
                if util.is_nested(name=name):
                    sequence_value = OrderedDict()
                    for inner_name, spec in self.values_spec[name].items():
                        sequence_value[inner_name] = tf.gather(
                            params=self.buffers[name][inner_name], indices=predecessor_indices
                        )
                else:
                    sequence_value = tf.gather(
                        params=self.buffers[name], indices=predecessor_indices
                    )
                sequence_values[n] = sequence_value

            for n, name in enumerate(initial_values):
                if util.is_nested(name=name):
                    initial_value = OrderedDict()
                    for inner_name, spec in self.values_spec[name].items():
                        initial_value[inner_name] = tf.gather(
                            params=self.buffers[name][inner_name], indices=initial_indices
                        )
                else:
                    initial_value = tf.gather(
                        params=self.buffers[name], indices=initial_indices
                    )
                initial_values[n] = initial_value

        # def body(lengths, sequence_values, initial_values):
        #     # Retrieve previous indices
        #     previous_indices = tf.math.mod(x=(indices - lengths), y=capacity)
        #     previous_values = self.retrieve(
        #         indices=previous_indices, values=(tuple(sequence_values) + tuple(initial_values))
        #     )

        #     # Overwrite initial values
        #     for name in initial_values:
        #         initial_values[name] = previous_values[name]

        #     # Concatenate sequence values
        #     for name, value, previous_value in util.zip_items(sequence_values, previous_values):
        #         if util.is_nested(name=name):
        #             for inner_name, value, previous_value in util.zip_items(value, previous_value):
        #                 previous_value = tf.expand_dims(input=previous_value, axis=1)
        #                 sequence_values[name][inner_name] = tf.concat(
        #                     values=(previous_value, value), axis=1
        #                 )
        #         else:
        #             previous_value = tf.expand_dims(input=previous_value, axis=1)
        #             sequence_values[name] = tf.concat(values=(previous_value, value), axis=1)

        #     # Increment lengths unless start of episode
        #     with tf.control_dependencies(control_inputs=util.flatten(xs=previous_values)):
        #         previous_indices = tf.math.mod(x=(previous_indices - one), y=capacity)
        #         terminal = self.retrieve(indices=previous_indices, values='terminal')
        #         x = tf.zeros_like(input=terminal, dtype=util.tf_dtype(dtype='long'))
        #         y = tf.ones_like(input=terminal, dtype=util.tf_dtype(dtype='long'))
        #         lengths += tf.where(condition=terminal, x=x, y=y)

        #     return lengths, sequence_values, initial_values

        # # Sequence lengths
        # lengths = tf.zeros_like(input=indices, dtype=util.tf_dtype(dtype='long'))

        # # Shape invariants
        # start_sequence_values = OrderedDict()
        # sequence_shapes = OrderedDict()
        # for name in sequence_values:
        #     if util.is_nested(name=name):
        #         start_sequence_values[name] = OrderedDict()
        #         sequence_shapes[name] = OrderedDict()
        #         for inner_name, spec in self.values_spec[name].items():
        #             start_sequence_values[name][inner_name] = tf.zeros(shape=((0, tf.shape(indices)[0]) + spec['shape']))
        #             shape = tf.TensorShape(dims=((None, None) + spec['shape']))
        #             sequence_shapes[name][inner_name] = shape
        #     else:
        #         start_sequence_values[name] = tf.zeros(shape=((0, tf.shape(indices)[0]) + self.values_spec[name]['shape']))
        #         shape = tf.TensorShape(dims=((None, None) + self.values_spec[name]['shape']))
        #         sequence_shapes[name] = shape
        # start_initial_values = OrderedDict()
        # initial_shapes = OrderedDict()
        # for name in initial_values:
        #     if util.is_nested(name=name):
        #         start_initial_values[name] = OrderedDict()
        #         initial_shapes[name] = OrderedDict()
        #         for inner_name, spec in self.values_spec[name].items():
        #             start_initial_values[name][inner_name] = tf.zeros(shape=((tf.shape(indices)[0],) + spec['shape']))
        #             shape = tf.TensorShape(dims=((None,) + spec['shape']))
        #             initial_shapes[name][inner_name] = shape
        #     else:
        #         start_initial_values[name] = tf.zeros(shape=((tf.shape(indices)[0],) + self.values_spec[name]['shape']))
        #         shape = tf.TensorShape(dims=((None,) + self.values_spec[name]['shape']))
        #         initial_shapes[name] = shape

        # # Retrieve predecessors
        # lengths, sequence_values, initial_values = self.while_loop(
        #     cond=util.tf_always_true, body=body,
        #     loop_vars=(lengths, start_sequence_values, start_initial_values),
        #     shape_invariants=(lengths.get_shape(), sequence_shapes, initial_shapes),
        #     back_prop=False, maximum_iterations=horizon
        # )

        # # Stop gradients
        # sequence_values = util.fmap(function=tf.stop_gradient, xs=sequence_values)
        # initial_values = util.fmap(function=tf.stop_gradient, xs=initial_values)

        if len(sequence_values) == 0:
            if is_single_initial_value:
                initial_values = initial_values[0]
            return lengths, initial_values

        elif len(initial_values) == 0:
            if is_single_sequence_value:
                sequence_values = sequence_values[0]
            return starts, lengths, sequence_values

        else:
            if is_single_sequence_value:
                sequence_values = sequence_values[0]
            if is_single_initial_value:
                initial_values = initial_values[0]
            return starts, lengths, sequence_values, initial_values

    def tf_successors(self, indices, horizon, sequence_values=(), final_values=()):
        if sequence_values == () and final_values == ():
            raise TensorforceError.unexpected()

        if isinstance(sequence_values, str):
            is_single_sequence_value = True
            sequence_values = [sequence_values]
        else:
            is_single_sequence_value = False
            sequence_values = list(sequence_values)
        if isinstance(final_values, str):
            is_single_final_value = True
            final_values = [final_values]
        else:
            is_single_final_value = False
            final_values = list(final_values)

        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        def body(lengths, successor_indices, mask):
            current_index = successor_indices[:, -1:]
            current_terminal = self.retrieve(indices=current_index, values='terminal')
            is_not_terminal = tf.math.logical_and(
                x=tf.math.logical_not(x=tf.math.greater(x=current_terminal, y=zero)),
                y=mask[:, -1:]
            )
            next_index = tf.math.mod(x=(current_index + one), y=capacity)
            successor_indices = tf.concat(values=(successor_indices, next_index), axis=1)
            mask = tf.concat(values=(mask, is_not_terminal), axis=1)
            is_not_terminal = tf.squeeze(input=is_not_terminal, axis=1)
            zeros = tf.zeros_like(input=is_not_terminal, dtype=util.tf_dtype(dtype='long'))
            ones = tf.ones_like(input=is_not_terminal, dtype=util.tf_dtype(dtype='long'))
            lengths += tf.where(condition=is_not_terminal, x=ones, y=zeros)
            return lengths, successor_indices, mask

        lengths = tf.ones_like(input=indices, dtype=util.tf_dtype(dtype='long'))
        successor_indices = tf.expand_dims(input=indices, axis=1)
        mask = tf.ones_like(input=successor_indices, dtype=util.tf_dtype(dtype='bool'))
        shape = tf.TensorShape(dims=((None, None)))

        lengths, successor_indices, mask = self.while_loop(
            cond=util.tf_always_true, body=body, loop_vars=(lengths, successor_indices, mask),
            shape_invariants=(lengths.get_shape(), shape, shape), back_prop=False,
            maximum_iterations=horizon
        )

        successor_indices = tf.reshape(tensor=successor_indices, shape=(-1,))
        mask = tf.reshape(tensor=mask, shape=(-1,))
        successor_indices = tf.boolean_mask(tensor=successor_indices, mask=mask, axis=0)

        assertion = tf.debugging.assert_greater_equal(
            x=tf.math.mod(x=(self.buffer_index - one - successor_indices), y=capacity), y=zero
        )

        with tf.control_dependencies(control_inputs=(assertion,)):
            starts = tf.math.cumsum(x=lengths, exclusive=True)
            ends = tf.math.cumsum(x=lengths) - one
            final_indices = tf.gather(params=successor_indices, indices=ends)

            for n, name in enumerate(sequence_values):
                if util.is_nested(name=name):
                    sequence_value = OrderedDict()
                    for inner_name, spec in self.values_spec[name].items():
                        sequence_value[inner_name] = tf.gather(
                            params=self.buffers[name][inner_name], indices=successor_indices
                        )
                else:
                    sequence_value = tf.gather(
                        params=self.buffers[name], indices=successor_indices
                    )
                sequence_values[n] = sequence_value

            for n, name in enumerate(final_values):
                if util.is_nested(name=name):
                    final_value = OrderedDict()
                    for inner_name, spec in self.values_spec[name].items():
                        final_value[inner_name] = tf.gather(
                            params=self.buffers[name][inner_name], indices=final_indices
                        )
                else:
                    final_value = tf.gather(
                        params=self.buffers[name], indices=final_indices
                    )
                final_values[n] = final_value

        # def body(lengths, sequence_values, final_values):
        #     # Retrieve next indices
        #     next_indices = tf.math.mod(x=(indices - lengths), y=capacity)
        #     next_values = self.retrieve(
        #         indices=next_indices, values=(tuple(sequence_values) + tuple(final_values))
        #     )

        #     # Overwrite final values
        #     for name in final_values:
        #         final_values[name] = next_values[name]

        #     # Concatenate sequence values
        #     for name, value, next_value in util.zip_items(sequence_values, next_values):
        #         if util.is_nested(name=name):
        #             for inner_name, value, next_value in util.zip_items(value, next_value):
        #                 next_value = tf.expand_dims(input=next_value, axis=1)
        #                 sequence_values[name][inner_name] = tf.concat(
        #                     values=(value, next_value), axis=1
        #                 )
        #         else:
        #             next_value = tf.expand_dims(input=next_value, axis=1)
        #             sequence_values[name] = tf.concat(values=(value, next_value), axis=1)

        #     # Increment lengths unless start of episode
        #     with tf.control_dependencies(control_inputs=util.flatten(xs=next_values)):
        #         next_indices = tf.math.mod(x=(next_indices - one), y=capacity)
        #         terminal = self.retrieve(indices=next_indices, values='terminal')
        #         x = tf.zeros_like(input=terminal, dtype=util.tf_dtype(dtype='long'))
        #         y = tf.ones_like(input=terminal, dtype=util.tf_dtype(dtype='long'))
        #         lengths += tf.where(condition=terminal, x=x, y=y)

        #     return lengths, sequence_values, final_values

        # # Sequence lengths
        # lengths = tf.zeros_like(input=indices, dtype=util.tf_dtype(dtype='long'))

        # # Shape invariants
        # start_sequence_values = OrderedDict()
        # sequence_shapes = OrderedDict()
        # for name in sequence_values:
        #     if util.is_nested(name=name):
        #         start_sequence_values[name] = OrderedDict()
        #         sequence_shapes[name] = OrderedDict()
        #         for inner_name, spec in self.values_spec[name].items():
        #             start_sequence_values[name][inner_name] = tf.zeros(shape=((0, tf.shape(indices)[0]) + spec['shape']))
        #             shape = tf.TensorShape(dims=((None, None) + spec['shape']))
        #             sequence_shapes[name][inner_name] = shape
        #     else:
        #         start_sequence_values[name] = tf.zeros(shape=((0, tf.shape(indices)[0]) + self.values_spec[name]['shape']))
        #         shape = tf.TensorShape(dims=((None, None) + self.values_spec[name]['shape']))
        #         sequence_shapes[name] = shape
        # start_final_values = OrderedDict()
        # final_shapes = OrderedDict()
        # for name in final_values:
        #     if util.is_nested(name=name):
        #         start_final_values[name] = OrderedDict()
        #         final_shapes[name] = OrderedDict()
        #         for inner_name, spec in self.values_spec[name].items():
        #             start_final_values[name][inner_name] = tf.zeros(shape=((tf.shape(indices)[0],) + spec['shape']))
        #             shape = tf.TensorShape(dims=((None,) + spec['shape']))
        #             final_shapes[name][inner_name] = shape
        #     else:
        #         start_final_values[name] = tf.zeros(shape=((tf.shape(indices)[0],) + self.values_spec[name]['shape']))
        #         shape = tf.TensorShape(dims=((None,) + self.values_spec[name]['shape']))
        #         final_shapes[name] = shape

        # # Retrieve predecessors
        # lengths, sequence_values, final_values = self.while_loop(
        #     cond=util.tf_always_true, body=body,
        #     loop_vars=(lengths, start_sequence_values, start_final_values),
        #     shape_invariants=(lengths.get_shape(), sequence_shapes, final_shapes),
        #     back_prop=False, maximum_iterations=horizon
        # )

        # # Stop gradients
        # sequence_values = util.fmap(function=tf.stop_gradient, xs=sequence_values)
        # final_values = util.fmap(function=tf.stop_gradient, xs=final_values)

        if len(sequence_values) == 0:
            if is_single_final_value:
                final_values = final_values[0]
            return lengths, final_values

        elif len(final_values) == 0:
            if is_single_sequence_value:
                sequence_values = sequence_values[0]
            return starts, lengths, sequence_values

        else:
            if is_single_sequence_value:
                sequence_values = sequence_values[0]
            if is_single_final_value:
                final_values = final_values[0]
            return starts, lengths, sequence_values, final_values
