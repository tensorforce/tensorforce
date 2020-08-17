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

from tensorforce.core import parameter_modules, SignatureDict, TensorSpec, tf_function, tf_util
from tensorforce.core.layers import Layer


class PreprocessingLayer(Layer):
    """
    Base class for preprocessing layers which require to be reset.

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, name=None, input_spec=None):
        super().__init__(name=name, input_spec=input_spec)

    def input_signature(self, *, function):
        if function == 'reset':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'reset':
            return SignatureDict(
                singleton=TensorSpec(type='bool', shape=()).signature(batched=False)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=0)
    def reset(self):
        raise NotImplementedError


class Clipping(Layer):
    """
    Clipping layer (specification key: `clipping`).

    Args:
        lower (parameter, float): Lower clipping value
            (<span style="color:#00C000"><b>default</b></span>: no lower bound).
        upper (parameter, float): Upper clipping value
            (<span style="color:#00C000"><b>default</b></span>: no upper bound).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, lower=None, upper=None, name=None, input_spec=None):
        super().__init__(name=name, input_spec=input_spec)

        if lower is None:
            assert upper is not None
            self.lower = None
        else:
            self.lower = self.submodule(
                name='lower', module=lower, modules=parameter_modules, dtype='float'
            )

        if upper is None:
            assert lower is not None
            self.upper = None
        else:
            self.upper = self.submodule(
                name='upper', module=upper, modules=parameter_modules, dtype='float'
            )

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    @tf_function(num_args=1)
    def apply(self, *, x):
        if self.lower is None:
            upper = self.upper.value()
            return tf.math.minimum(x=x, y=upper)
        elif self.upper is None:
            lower = self.lower.value()
            return tf.math.maximum(x=x, y=lower)
        else:
            lower = self.lower.value()
            upper = self.upper.value()
            assertions = list()
            if self.config.create_tf_assertions:
                assertions.append(tf.debugging.assert_greater_equal(
                    x=upper, y=lower, message="Incompatible lower and upper clipping bound."
                ))
            with tf.control_dependencies(control_inputs=assertions):
                return tf.clip_by_value(t=x, clip_value_min=lower, clip_value_max=upper)


class Deltafier(PreprocessingLayer):
    """
    Deltafier layer computing the difference between the current and the previous input; can only
    be used as preprocessing layer (specification key: `deltafier`).

    Args:
        concatenate (False | int >= 0): Whether to concatenate instead of replace deltas with
            input, and if so, concatenation axis
            (<span style="color:#00C000"><b>default</b></span>: false).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, concatenate=False, name=None, input_spec=None):
        self.concatenate = concatenate

        super().__init__(name=name, input_spec=input_spec)

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    def output_spec(self):
        output_spec = super().output_spec()

        if self.concatenate is not False:
            output_spec.shape = tuple(
                2 * dims if axis == self.concatenate else dims
                for axis, dims in enumerate(output_spec.shape)
            )

        return output_spec

    def initialize(self):
        super().initialize()

        self.has_previous = self.variable(
            name='has-previous', spec=TensorSpec(type='bool', shape=()), initializer='zeros',
            is_trainable=False, is_saved=False
        )

        self.previous = self.variable(
            name='previous', spec=TensorSpec(type='float', shape=((1,) + self.input_spec.shape)),
            initializer='zeros', is_trainable=False, is_saved=False
        )

    @tf_function(num_args=0)
    def reset(self):
        false = tf_util.constant(value=False, dtype='bool')
        assignment = self.has_previous.assign(value=false, read_value=False)
        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf_util.identity(input=false)

    @tf_function(num_args=1)
    def apply(self, *, x):
        assertions = list()
        if self.config.create_tf_assertions:
            assertions.append(tf.debugging.assert_less_equal(
                x=tf.shape(input=x)[0], y=1,
                message="Deltafier preprocessor currently not compatible with batched Agent.act."
            ))

        # TODO: hack for empty batch (for self.previous.assign below)
        extended = tf.concat(values=(self.previous, x), axis=0)

        def first_delta():
            assignment = self.has_previous.assign(
                value=tf_util.constant(value=True, dtype='bool'), read_value=False
            )
            with tf.control_dependencies(control_inputs=(assignment,)):
                return tf.concat(values=(tf.zeros_like(input=x[:1]), x[1:] - x[:-1]), axis=0)

        def later_delta():
            return x - extended[:-1]

        with tf.control_dependencies(control_inputs=assertions):
            empty_batch = tf.math.equal(x=tf.shape(input=x)[0], y=0)
            pred = tf.math.logical_or(x=self.has_previous, y=empty_batch)
            delta = tf.cond(pred=pred, true_fn=later_delta, false_fn=first_delta)

            assignment = self.previous.assign(value=extended[-1:], read_value=False)

        with tf.control_dependencies(control_inputs=(assignment,)):
            if self.concatenate is False:
                return tf_util.identity(input=delta)
            else:
                return tf.concat(values=(x, delta), axis=(self.concatenate + 1))


class Image(Layer):
    """
    Image preprocessing layer (specification key: `image`).

    Args:
        height (int): Height of resized image
            (<span style="color:#00C000"><b>default</b></span>: no resizing or relative to width).
        width (int): Width of resized image
            (<span style="color:#00C000"><b>default</b></span>: no resizing or relative to height).
        grayscale (bool | iter[float]): Turn into grayscale image, optionally using given weights
            (<span style="color:#00C000"><b>default</b></span>: false).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, height=None, width=None, grayscale=False, name=None, input_spec=None):
        self.height = height
        self.width = width
        self.grayscale = grayscale

        super().__init__(name=name, input_spec=input_spec)

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0, 0, 0))

    def output_spec(self):
        output_spec = super().output_spec()

        if self.height is not None:
            if self.width is None:
                self.width = round(self.height * output_spec.shape[1] / output_spec.shape[0])
            output_spec.shape = (self.height, self.width, output_spec.shape[2])
        elif self.width is not None:
            self.height = round(self.width * output_spec.shape[0] / output_spec.shape[1])
            output_spec.shape = (self.height, self.width, output_spec.shape[2])

        if not isinstance(self.grayscale, bool) or self.grayscale:
            output_spec.shape = output_spec.shape[:2] + (1,)

        return output_spec

    @tf_function(num_args=1)
    def apply(self, *, x):
        if self.height is not None:
            x = tf.image.resize(images=x, size=(self.height, self.width))

        if not isinstance(self.grayscale, bool):
            weights = tf_util.constant(
                value=self.grayscale, dtype='float', shape=(1, 1, 1, len(self.grayscale))
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
        length (int > 0): Number of inputs to concatenate
            (<span style="color:#C00000"><b>required</b></span>).
        axis (int >= 0): Concatenation axis, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: last axis).
        concatenate (bool): Whether to concatenate inputs at given axis, otherwise introduce new
            sequence axis
            (<span style="color:#00C000"><b>default</b></span>: true).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, length, axis=-1, concatenate=True, name=None, input_spec=None):
        assert length > 1
        self.length = length
        self.axis = axis
        self.concatenate = concatenate

        super().__init__(name=name, input_spec=input_spec)

    def output_spec(self):
        output_spec = super().output_spec()

        if self.concatenate:
            if self.axis == -1:
                self.axis = len(output_spec.shape) - 1
            output_spec.shape = tuple(
                self.length * dims if axis == self.axis else dims
                for axis, dims in enumerate(output_spec.shape)
            )

        else:
            if self.axis == -1:
                self.axis = len(output_spec.shape)
            shape = output_spec.shape
            output_spec.shape = shape[:self.axis] + (self.length,) + shape[self.axis:]

        return output_spec

    def initialize(self):
        super().initialize()

        self.has_previous = self.variable(
            name='has-previous', spec=TensorSpec(type='bool', shape=()), initializer='zeros',
            is_trainable=False, is_saved=False
        )

        shape = self.input_spec.shape
        if self.concatenate:
            shape = (1,) + shape[:self.axis] + (shape[self.axis] * (self.length - 1),) + \
                shape[self.axis + 1:]
        else:
            shape = (1,) + shape[:self.axis] + (self.length - 1,) + shape[self.axis:]
        self.previous = self.variable(
            name='previous', spec=TensorSpec(type='float', shape=shape), initializer='zeros',
            is_trainable=False, is_saved=False
        )

    @tf_function(num_args=0)
    def reset(self):
        false = tf_util.constant(value=False, dtype='bool')
        assignment = self.has_previous.assign(value=false, read_value=False)
        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf_util.identity(input=false)

    @tf_function(num_args=1)
    def apply(self, *, x):
        assertions = list()
        if self.config.create_tf_assertions:
            assertions.append(tf.debugging.assert_less_equal(
                x=tf.shape(input=x)[0], y=1,
                message="Sequence preprocessor currently not compatible with batched Agent.act."
            ))

        with tf.control_dependencies(control_inputs=assertions):

            def empty_batch():
                if self.concatenate:
                    current = x
                else:
                    current = tf.expand_dims(input=x, axis=(self.axis + 1))
                multiples = tuple(
                    self.length if dims == self.axis + 1 else 1
                    for dims in range(self.output_spec().rank + 1)
                )
                return tf.tile(input=current, multiples=multiples)

            def not_empty_batch():

                def first_timestep():
                    assignment = self.has_previous.assign(
                        value=tf_util.constant(value=True, dtype='bool'), read_value=False
                    )
                    with tf.control_dependencies(control_inputs=(assignment,)):
                        if self.concatenate:
                            current = x
                        else:
                            current = tf.expand_dims(input=x, axis=(self.axis + 1))
                        multiples = tuple(
                            self.length if dims == self.axis + 1 else 1
                            for dims in range(self.output_spec().rank + 1)
                        )
                        return tf.tile(input=current, multiples=multiples)

                def other_timesteps():
                    if self.concatenate:
                        current = x
                    else:
                        current = tf.expand_dims(input=x, axis=(self.axis + 1))
                    return tf.concat(values=(self.previous, current), axis=(self.axis + 1))

                xs = tf.cond(
                    pred=self.has_previous, true_fn=other_timesteps, false_fn=first_timestep
                )

                if self.concatenate:
                    begin = tuple(
                        self.input_spec.shape[dims - 1] if dims == self.axis + 1 else 0
                        for dims in range(self.output_spec().rank + 1)
                    )
                else:
                    begin = tuple(
                        1 if dims == self.axis + 1 else 0
                        for dims in range(self.output_spec().rank + 1)
                    )
                assignment = self.previous.assign(
                    value=tf.slice(input_=xs, begin=begin, size=self.previous.shape),
                    read_value=False
                )

                with tf.control_dependencies(control_inputs=(assignment,)):
                    return tf_util.identity(input=xs)

            is_empty_batch = tf.math.equal(x=tf.shape(input=x)[0], y=0)
            return tf.cond(pred=is_empty_batch, true_fn=empty_batch, false_fn=not_empty_batch)
