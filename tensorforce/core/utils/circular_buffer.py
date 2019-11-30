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

import tensorflow as tf

from tensorforce import util
from tensorforce.core import Module


class CircularBuffer(Module):
    """
    Circular buffer.

    Args:
        name (string): Buffer name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        capacity (int > 0): Buffer capacity
            (<span style="color:#C00000"><b>required</b></span>).
        values_spec (specification): Values specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        return_overwritten (bool): Whether to return overwritten values
            (<span style="color:#00C000"><b>default</b></span>: false).
        initializers (dict[values]): Buffer initializers
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, capacity, values_spec, return_overwritten=False, initializers=None,
        device=None, summary_labels=None
    ):
        super().__init__(name=name, device=device, summary_labels=summary_labels)

        self.values_spec = values_spec
        self.capacity = capacity
        self.return_overwritten = return_overwritten
        self.initializers = OrderedDict() if initializers is None else initializers

    def tf_initialize(self):
        super().tf_initialize()

        # Value buffers
        self.buffers = OrderedDict()
        for name, spec in self.values_spec.items():
            if util.is_nested(name=name):
                self.buffers[name] = OrderedDict()
                for inner_name, spec in spec.items():
                    shape = (self.capacity,) + spec['shape']
                    initializer = self.initializers.get(inner_name, 'zeros')
                    self.buffers[name][inner_name] = self.add_variable(
                        name=(inner_name + '-buffer'), dtype=spec['type'], shape=shape,
                        is_trainable=False, initializer=initializer
                    )
            else:
                shape = (self.capacity,) + spec['shape']
                initializer = self.initializers.get(name, 'zeros')
                self.buffers[name] = self.add_variable(
                    name=(name + '-buffer'), dtype=spec['type'], shape=shape, is_trainable=False,
                    initializer=initializer
                )

        # Buffer index (modulo capacity, next index to write to)
        self.buffer_index = self.add_variable(
            name='buffer-index', dtype='long', shape=(), is_trainable=False, initializer='zeros'
        )

    def tf_reset(self):
        # Constants
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        if not self.return_overwritten:
            # Reset buffer index
            assignment = self.buffer_index.assign(value=zero, read_value=False)

            # Return no-op
            with tf.control_dependencies(control_inputs=(assignment,)):
                return util.no_operation()

        # Overwritten buffer indices
        num_values = tf.minimum(x=self.buffer_index, y=capacity)
        indices = tf.range(start=(self.buffer_index - num_values), limit=self.buffer_index)
        indices = tf.math.mod(x=indices, y=capacity)

        # Get overwritten values
        values = OrderedDict()
        for name, buffer in self.buffers.items():
            if util.is_nested(name=name):
                values[name] = OrderedDict()
                for inner_name, buffer in buffer.items():
                    values[name][inner_name] = tf.gather(params=buffer, indices=indices)
            else:
                values[name] = tf.gather(params=buffer, indices=indices)

        # Reset buffer index
        with tf.control_dependencies(control_inputs=util.flatten(xs=values)):
            assignment = self.buffer_index.assign(value=zero, read_value=False)

        # Return overwritten values
        with tf.control_dependencies(control_inputs=(assignment,)):
            return util.fmap(function=util.identity_operation, xs=values)

    def tf_enqueue(self, **values):
        # Constants
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        capacity = tf.constant(value=self.capacity, dtype=util.tf_dtype(dtype='long'))

        # Get number of values
        for value in values.values():
            if not isinstance(value, dict):
                break
            elif len(value) > 0:
                value = next(iter(value.values()))
                break
        if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
            num_values = tf.shape(input=value, out_type=util.tf_dtype(dtype='long'))[0]
        else:
            num_values = tf.dtypes.cast(
                x=tf.shape(input=value)[0], dtype=util.tf_dtype(dtype='long')
            )

        # Check whether instances fit into buffer
        assertion = tf.debugging.assert_less_equal(x=num_values, y=capacity)

        if self.return_overwritten:
            # Overwritten buffer indices
            with tf.control_dependencies(control_inputs=(assertion,)):
                start = tf.maximum(x=self.buffer_index, y=capacity)
                limit = tf.maximum(x=(self.buffer_index + num_values), y=capacity)
                num_overwritten = limit - start
                indices = tf.range(start=start, limit=limit)
                indices = tf.math.mod(x=indices, y=capacity)

            # Get overwritten values
            with tf.control_dependencies(control_inputs=(indices,)):
                overwritten_values = OrderedDict()
                for name, buffer in self.buffers.items():
                    if util.is_nested(name=name):
                        overwritten_values[name] = OrderedDict()
                        for inner_name, buffer in buffer.items():
                            overwritten_values[name][inner_name] = tf.gather(
                                params=buffer, indices=indices
                            )
                    else:
                        overwritten_values[name] = tf.gather(params=buffer, indices=indices)

        else:
            overwritten_values = (assertion,)

        # Buffer indices to (over)write
        with tf.control_dependencies(control_inputs=util.flatten(xs=overwritten_values)):
            indices = tf.range(start=self.buffer_index, limit=(self.buffer_index + num_values))
            indices = tf.math.mod(x=indices, y=capacity)
            indices = tf.expand_dims(input=indices, axis=1)

        # Write new values
        with tf.control_dependencies(control_inputs=(indices,)):
            assignments = list()
            for name, buffer in self.buffers.items():
                if util.is_nested(name=name):
                    for inner_name, buffer in buffer.items():
                        assignment = buffer.scatter_nd_update(
                            indices=indices, updates=values[name][inner_name]
                        )
                        assignments.append(assignment)
                else:
                    assignment = buffer.scatter_nd_update(indices=indices, updates=values[name])
                    assignments.append(assignment)

        # Increment buffer index
        with tf.control_dependencies(control_inputs=assignments):
            assignment = self.buffer_index.assign_add(delta=num_values, read_value=False)

        # Return overwritten values or no-op
        with tf.control_dependencies(control_inputs=(assignment,)):
            if self.return_overwritten:
                any_overwritten = tf.math.greater(x=num_overwritten, y=zero)
                overwritten_values = util.fmap(
                    function=util.identity_operation, xs=overwritten_values
                )
                return any_overwritten, overwritten_values
            else:
                return util.no_operation()
