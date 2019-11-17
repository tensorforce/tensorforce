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
from tensorforce.core import parameter_modules
from tensorforce.core.layers import Layer


class PreprocessingLayer(Layer):

    def tf_reset(self):
        raise NotImplementedError


class Clipping(Layer):
    """
    Clipping layer (specification key: `clipping`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        upper (parameter, float): Upper clipping value
            (<span style="color:#C00000"><b>required</b></span>).
        lower (parameter, float): Lower clipping value
            (<span style="color:#00C000"><b>default</b></span>: negative upper value).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, upper, lower=None, input_spec=None, summary_labels=None):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

        self.upper = self.add_module(
            name='upper', module=lower, modules=parameter_modules, dtype='float'
        )

        if lower is None:
            self.lower = None
        else:
            self.lower = self.add_module(
                name='lower', module=lower, modules=parameter_modules, dtype='float'
            )

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def tf_apply(self, x):
        upper = self.upper.value()
        if self.lower is None:
            lower = -upper
        else:
            lower = self.lower.value()

        assertion = tf.debugging.assert_greater_equal(x=upper, y=lower)

        with tf.control_dependencies(control_inputs=(assertion,)):
            return tf.clip_by_value(t=x, clip_value_min=lower, clip_value_max=upper)


class Deltafier(PreprocessingLayer):
    """
    Deltafier layer computing the difference between the current and the previous input; can only
    be used as preprocessing layer (specification key: `deltafier`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        concatenate (False | int >= 0): Whether to concatenate instead of replace deltas with
            input, and if so, concatenation axis
            (<span style="color:#00C000"><b>default</b></span>: false).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, concatenate=False, input_spec=None, summary_labels=None):
        self.concatenate = concatenate

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def get_output_spec(self, input_spec):
        if self.concatenate is not False:
            input_spec['shape'] = tuple(
                2 * dims if axis == self.concatenate else dims
                for axis, dims in enumerate(input_spec['shape'])
            )

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        self.has_previous = self.add_variable(
            name='has-previous', dtype='bool', shape=(), is_trainable=False, initializer='zeros'
        )

        self.previous = self.add_variable(
            name='previous', dtype='float', shape=((1,) + self.input_spec['shape']),
            is_trainable=False, initializer='zeros'
        )

    def tf_reset(self):
        assignment = self.has_previous.assign(
            value=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')), read_value=False
        )
        return assignment

    def tf_apply(self, x):

        def first_delta():
            assignment = self.has_previous.assign(
                value=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')), read_value=False
            )
            with tf.control_dependencies(control_inputs=(assignment,)):
                return tf.concat(values=(tf.zeros_like(input=x[:1]), x[1:] - x[:-1]), axis=0)  # dtype=util.tf_dtype(dtype='???'))

        def later_delta():
            return x - tf.concat(values=(self.previous, x[:-1]), axis=0)

        delta = self.cond(pred=self.has_previous, true_fn=later_delta, false_fn=first_delta)

        assignment = self.previous.assign(value=x[-1:], read_value=False)

        with tf.control_dependencies(control_inputs=(assignment,)):
            if self.concatenate is False:
                return util.identity_operation(x=delta)
            else:
                return tf.concat(values=(x, delta), axis=(self.concatenate + 1))


class Image(Layer):
    """
    Image preprocessing layer (specification key: `image`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        height (int): Height of resized image
            (<span style="color:#00C000"><b>default</b></span>: no resizing or relative to width).
        width (int): Width of resized image
            (<span style="color:#00C000"><b>default</b></span>: no resizing or relative to height).
        grayscale (bool | iter[float]): Turn into grayscale image, optionally using given weights
            (<span style="color:#00C000"><b>default</b></span>: false).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, height=None, width=None, grayscale=False, input_spec=None, summary_labels=None
    ):
        self.height = height
        self.width = width
        self.grayscale = grayscale

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

    def default_input_spec(self):
        return dict(type='float', shape=(0, 0, 0))

    def get_output_spec(self, input_spec):
        if self.height is not None:
            if self.width is None:
                self.width = round(self.height * input_spec['shape'][1] / input_spec['shape'][0])
            input_spec['shape'] = (self.height, self.width, input_spec['shape'][2])
        elif self.width is not None:
            self.height = round(self.width * input_spec['shape'][0] / input_spec['shape'][1])
            input_spec['shape'] = (self.height, self.width, input_spec['shape'][2])

        if not isinstance(self.grayscale, bool) or self.grayscale:
            input_spec['shape'] = input_spec['shape'][:2] + (1,)

        return input_spec

    def tf_apply(self, x):
        if self.height is not None:
            x = tf.image.resize(images=x, size=(self.height, self.width))

        if not isinstance(self.grayscale, bool):
            weights = tf.constant(
                value=self.grayscale, dtype=util.tf_dtype(dtype='float'),
                shape=(1, 1, 1, len(self.grayscale))
            )
            x = tf.reduce_sum(input_tensor=(x * weights), axis=3, keepdims=True)
        elif self.grayscale:
            x = tf.image.rgb_to_grayscale(images=x)

        return x


class Sequence(PreprocessingLayer):
    """
    Sequence layer stacking the current and previous inputs; can only be used as preprocessing
    layer (specification key: `sequence`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        length (int > 0): Number of inputs to concatenate
            (<span style="color:#C00000"><b>required</b></span>).
        axis (int >= 0): Concatenation axis, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: last axis).
        concatenate (bool): Whether to concatenate inputs at given axis, otherwise introduce new
            sequence axis
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, length, axis=-1, concatenate=True, input_spec=None, summary_labels=None
    ):
        self.length = length
        self.axis = axis
        self.concatenate = concatenate

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        if self.concatenate:
            if self.axis == -1:
                self.axis = len(input_spec['shape']) - 1
            input_spec['shape'] = tuple(
                self.length * dims if axis == self.axis else dims
                for axis, dims in enumerate(input_spec['shape'])
            )

        else:
            if self.axis == -1:
                self.axis = len(input_spec['shape'])
            shape = input_spec['shape']
            input_spec['shape'] = shape[:self.axis] + (self.length,) + shape[self.axis:]

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        self.has_previous = self.add_variable(
            name='has-previous', dtype='bool', shape=(), is_trainable=False, initializer='zeros'
        )

        self.previous = self.add_variable(
            name='previous', dtype='float', shape=((self.length - 1,) + self.input_spec['shape']),
            is_trainable=False, initializer='zeros'
        )

    def tf_reset(self):
        assignment = self.has_previous.assign(
            value=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')), read_value=False
        )
        return assignment

    def tf_apply(self, x):

        def first_sequence():
            assignment = self.has_previous.assign(
                value=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')), read_value=False
            )
            with tf.control_dependencies(control_inputs=(assignment,)):
                if self.concatenate:
                    current = x
                else:
                    current = tf.expand_dims(input=x, axis=(self.axis + 1))
                multiples = tuple(
                    self.length if dims == self.axis + 1 else 1
                    for dims in range(util.rank(x=current))
                )
                return tf.tile(input=x, multiples=multiples)

        def later_sequence():
            tf.concat(values=(self.previous, x))
            if self.concatenate:
                current = x
            else:
                current = tf.expand_dims(input=x, axis=(self.axis + 1))
            return tf.concat(values=(self.previous, current), axis=(self.axis + 1))

        sequence = self.cond(pred=self.has_previous, true_fn=later_sequence, false_fn=first_sequence)

        assignment = self.previous.assign(
            value=tf.concat(values=(self.previous, x), axis=0)[-self.length + 1:], read_value=False
        )

        with tf.control_dependencies(control_inputs=(assignment,)):
            return util.identity_operation(x=sequence)
