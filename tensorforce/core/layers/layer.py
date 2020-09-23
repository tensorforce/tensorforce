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

from tensorforce import TensorforceError, util
import tensorforce.core
from tensorforce.core import ArrayDict, Module, parameter_modules, SignatureDict, TensorSpec, \
    TensorsSpec, tf_function, tf_util
from tensorforce.core.parameters import Parameter


class Layer(Module):
    """
    Base class for neural network layers.

    Args:
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    _TF_MODULE_IGNORED_PROPERTIES = Module._TF_MODULE_IGNORED_PROPERTIES | {'_REGISTERED_LAYERS'}

    # _REGISTERED_LAYERS  # Initialized as part of model.__init__()

    def __init__(self, *, l2_regularization=None, name=None, input_spec=None):
        super().__init__(l2_regularization=l2_regularization, name=name)

        Layer._REGISTERED_LAYERS[self.name] = self

        self.input_spec = self.default_input_spec()
        if not isinstance(self.input_spec, TensorSpec):
            raise TensorforceError.unexpected()

        self.input_spec = self.input_spec.unify(
            other=input_spec, name=(self.__class__.__name__ + ' input')
        )

    def default_input_spec(self):
        return TensorSpec(type=None, shape=None, overwrite=True)

    def output_spec(self):
        return self.input_spec.copy(overwrite=True)

    def submodule(self, *args, **kwargs):
        layer = super().submodule(*args, **kwargs)

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(name='layer', argument='submodule', dtype=type(layer))

        return layer

    def input_signature(self, *, function):
        if function == 'apply':
            return SignatureDict(x=self.input_spec.signature(batched=True))

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'apply':
            return SignatureDict(singleton=self.output_spec().signature(batched=True))

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=1)
    def apply(self, *, x):
        return x


class MultiInputLayer(Layer):
    """
    Base class for multi-input layers.

    Args:
        tensors (iter[string]): Names of tensors to retrieve, either state names or previously
            registered tensors
            (<span style="color:#C00000"><b>required</b></span>).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, tensors, l2_regularization=None, name=None, input_spec=None):
        super(Layer, self).__init__(l2_regularization=l2_regularization, name=name)

        Layer._REGISTERED_LAYERS[self.name] = self

        if not util.is_iterable(x=tensors):
            raise TensorforceError.type(
                name='MultiInputLayer', argument='tensors', dtype=type(tensors)
            )
        elif len(tensors) == 0:
            raise TensorforceError.value(
                name='MultiInputLayer', argument='tensors', value=tensors, hint='zero length'
            )

        self.tensors = tuple(tensors)

        self.input_spec = self.default_input_spec()
        if not isinstance(self.input_spec, TensorsSpec):
            raise TensorforceError.unexpected()

        self.input_spec = self.input_spec.unify(other=input_spec)

    def default_input_spec(self):
        return TensorsSpec(
            ((tensor, TensorSpec(type=None, shape=None, overwrite=True)) for tensor in self.tensors)
        )

    def output_spec(self):
        return TensorSpec(type=None, shape=None, overwrite=True)

    @tf_function(num_args=1)
    def apply(self, *, x):
        raise NotImplementedError


class NondeterministicLayer(Layer):
    """
    Base class for nondeterministic layers.

    Args:
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def input_signature(self, *, function):
        if function == 'apply':
            return SignatureDict(
                x=self.input_spec.signature(batched=True),
                deterministic=TensorSpec(type='bool', shape=()).signature(batched=False)
            )

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=2, overwrites_signature=True)
    def apply(self, *, x, deterministic):
        raise NotImplementedError


class Register(Layer):
    """
    Tensor retrieval layer, which is useful when defining more complex network architectures which
    do not follow the sequential layer-stack pattern, for instance, when handling multiple inputs
    (specification key: `register`).

    Args:
        tensor (string): Name under which tensor will be registered
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, tensor, name=None, input_spec=None):
        super().__init__(name=name, input_spec=input_spec)

        if not isinstance(tensor, str):
            raise TensorforceError.type(name='register', argument='tensor', dtype=type(tensor))

        self.tensor = tensor


class Retrieve(MultiInputLayer):
    """
    Tensor retrieval layer, which is useful when defining more complex network architectures which
    do not follow the sequential layer-stack pattern, for instance, when handling multiple inputs
    (specification key: `retrieve`).

    Args:
        tensors (iter[string]): Names of tensors to retrieve, either state names or previously
            registered tensors
            (<span style="color:#C00000"><b>required</b></span>).
        aggregation ('concat' | 'product' | 'stack' | 'sum'): Aggregation type in case of multiple
            tensors
            (<span style="color:#00C000"><b>default</b></span>: 'concat').
        axis (int >= 0): Aggregation axis, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: 0).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, tensors, aggregation='concat', axis=0, name=None, input_spec=None):
        super().__init__(tensors=tensors, name=name, input_spec=input_spec)

        if aggregation not in ('concat', 'product', 'stack', 'sum'):
            raise TensorforceError.value(
                name='retrieve', argument='aggregation', value=aggregation,
                hint='not in {concat,product,stack,sum}'
            )

        self.aggregation = aggregation
        self.axis = axis

    def output_spec(self):
        if len(self.tensors) == 1:
            return self.input_spec[self.tensors[0]]

        # Get tensor types and shapes
        dtypes = list()
        shapes = list()
        for spec in self.input_spec.values():
            dtypes.append(spec.type)
            shapes.append(spec.shape)

        # Check tensor types
        if all(dtype == dtypes[0] for dtype in dtypes):
            dtype = dtypes[0]
        else:
            raise TensorforceError.value(name='retrieve', argument='tensor types', value=dtypes)

        if self.aggregation == 'concat':
            if any(len(shape) != len(shapes[0]) for shape in shapes):
                raise TensorforceError.value(
                    name='retrieve', argument='tensor shapes', value=shapes
                )
            elif any(
                shape[n] != shapes[0][n] for shape in shapes for n in range(len(shape))
                if n != self.axis
            ):
                raise TensorforceError.value(
                    name='retrieve', argument='tensor shapes', value=shapes
                )
            shape = tuple(
                sum(shape[n] for shape in shapes) if n == self.axis else shapes[0][n]
                for n in range(len(shapes[0]))
            )

        elif self.aggregation == 'stack':
            if any(len(shape) != len(shapes[0]) for shape in shapes):
                raise TensorforceError.value(
                    name='retrieve', argument='tensor shapes', value=shapes
                )
            elif any(shape[n] != shapes[0][n] for shape in shapes for n in range(len(shape))):
                raise TensorforceError.value(
                    name='retrieve', argument='tensor shapes', value=shapes
                )
            shape = tuple(
                len(shapes) if n == self.axis else shapes[0][n - int(n > self.axis)]
                for n in range(len(shapes[0]) + 1)
            )

        else:
            # Check and unify tensor shapes
            for shape in shapes:
                if len(shape) != len(shapes[0]):
                    raise TensorforceError.value(
                        name='retrieve', argument='tensor shapes', value=shapes
                    )
                if any(x != y and x != 1 and y != 1 for x, y in zip(shape, shapes[0])):
                    raise TensorforceError.value(
                        name='retrieve', argument='tensor shapes', value=shapes
                    )
            shape = tuple(max(shape[n] for shape in shapes) for n in range(len(shapes[0])))

        # TODO: Missing num_values, min/max_value
        return TensorSpec(type=dtype, shape=shape)

    @tf_function(num_args=1)
    def apply(self, *, x):
        if len(self.tensors) == 1:
            return x[self.tensors[0]]

        x = list(x.values())

        shape = self.output_spec().shape
        for n, tensor in enumerate(x):
            for axis in range(tf_util.rank(x=tensor), len(shape)):
                tensor = tf.expand_dims(input=tensor, axis=axis)
            x[n] = tensor

        if self.aggregation == 'concat':
            x = tf.concat(values=x, axis=(self.axis + 1))

        elif self.aggregation == 'product':
            x = tf.stack(values=x, axis=(self.axis + 1))
            x = tf.reduce_prod(input_tensor=x, axis=(self.axis + 1))

        elif self.aggregation == 'stack':
            x = tf.stack(values=x, axis=(self.axis + 1))

        elif self.aggregation == 'sum':
            x = tf.stack(values=x, axis=(self.axis + 1))
            x = tf.reduce_sum(input_tensor=x, axis=(self.axis + 1))

        return x


class Reuse(Layer):
    """
    Reuse layer (specification key: `reuse`).

    Args:
        layer (string): Name of a previously defined layer
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    # _TF_MODULE_IGNORED_PROPERTIES = Module._TF_MODULE_IGNORED_PROPERTIES | {'reused_layer'}

    def __init__(self, *, layer, name=None, input_spec=None):
        if layer not in Layer._REGISTERED_LAYERS:
            raise TensorforceError.value(name='reuse', argument='layer', value=layer)

        self.layer = layer

        super().__init__(name=name, input_spec=input_spec, l2_regularization=0.0)

    @property
    def reused_layer(self):
        return Layer._REGISTERED_LAYERS[self.layer]

    def default_input_spec(self):
        return self.reused_layer.input_spec.copy()

    def output_spec(self):
        return self.reused_layer.output_spec()

    @tf_function(num_args=1)
    def apply(self, *, x):
        return self.reused_layer.apply(x=x)

    # TODO: other Module functions?
    def get_available_summaries(self):
        summaries = super().get_available_summaries()
        summaries.update(self.reused_layer.get_available_summaries())
        return sorted(summaries)


class StatefulLayer(Layer):  # TODO: WeaklyStatefulLayer ?
    """
    Base class for stateful layers, i.e. layers which update an internal state for on-policy calls.

    Args:
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    @tf_function(num_args=1)
    def apply(self, *, x, independent):
        raise NotImplementedError


class TemporalLayer(Layer):
    """
    Base class for temporal layers, i.e. layers whose output depends on previous states.

    Args:
        temporal_processing ('cumulative' | 'iterative'): Temporal processing type
            (<span style="color:#C00000"><b>required</b></span>).
        horizon (parameter, int >= 0): Past horizon
            (<span style="color:#C00000"><b>required</b></span>).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
        kwargs: Additional arguments for potential parent class.
    """

    def __init__(
        self, *, temporal_processing, horizon, l2_regularization=None, name=None, input_spec=None,
        **kwargs
    ):
        if temporal_processing not in ('cumulative', 'iterative'):
            raise TensorforceError.value(
                name='temporal-layer', argument='temporal_processing', value=temporal_processing,
                hint='not in {cumulative,iterative}'
            )
        self.temporal_processing = temporal_processing

        super().__init__(
            l2_regularization=l2_regularization, name=name, input_spec=input_spec, **kwargs
        )

        if self.temporal_processing == 'cumulative' and len(self.internals_spec) > 0:
            raise TensorforceError.invalid(
                name='temporal-layer', argument='temporal_processing', expected='iterative',
                condition='num internals > 0'
            )

        if horizon is None:
            horizon = 0
        self.horizon = self.submodule(
            name='horizon', module=horizon, modules=parameter_modules, is_trainable=False,
            dtype='int', min_value=0
        )

    @property
    def internals_spec(self):
        return TensorsSpec()

    def internals_init(self):
        return ArrayDict()

    def max_past_horizon(self, *, on_policy):
        if self.temporal_processing == 'iterative' and on_policy:
            return 0
        else:
            return self.horizon.max_value()

    def input_signature(self, *, function):
        if function == 'apply':
            assert len(self.internals_spec) == 0 or self.temporal_processing == 'iterative'
            return SignatureDict(
                x=self.input_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True)
            )

        elif function == 'cumulative_apply':
            assert self.temporal_processing == 'cumulative'
            cumulative_input_spec = self.input_spec.copy()
            cumulative_input_spec.shape = (None,) + cumulative_input_spec.shape
            return SignatureDict(
                xs=cumulative_input_spec.signature(batched=True),
                lengths=TensorSpec(type='int', shape=()).signature(batched=True)
            )

        elif function == 'iterative_apply':
            assert self.temporal_processing == 'iterative'
            return SignatureDict(
                x=self.input_spec.signature(batched=True),
                internals=self.internals_spec.signature(batched=True)
            )

        elif function == 'iterative_body':
            assert self.temporal_processing == 'iterative'
            return SignatureDict(
                x=self.input_spec.signature(batched=True),
                indices=TensorSpec(type='int', shape=()).signature(batched=True),
                remaining=TensorSpec(type='int', shape=()).signature(batched=True),
                current_x=self.output_spec().signature(batched=True),
                current_internals=self.internals_spec.signature(batched=True)
            )

        elif function == 'past_horizon':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'apply':
            if self.temporal_processing == 'cumulative':
                return SignatureDict(singleton=self.output_spec().signature(batched=True))
            elif self.temporal_processing == 'iterative':
                return SignatureDict(
                    x=self.output_spec().signature(batched=True),
                    internals=self.internals_spec.signature(batched=True)
                )

        elif function == 'cumulative_apply':
            assert self.temporal_processing == 'cumulative'
            return SignatureDict(singleton=self.output_spec().signature(batched=True))

        elif function == 'iterative_apply':
            assert self.temporal_processing == 'iterative'
            return SignatureDict(
                x=self.output_spec().signature(batched=True),
                internals=self.internals_spec.signature(batched=True)
            )

        elif function == 'iterative_body':
            assert self.temporal_processing == 'iterative'
            return SignatureDict(
                x=self.input_spec.signature(batched=True),
                indices=TensorSpec(type='int', shape=()).signature(batched=True),
                remaining=TensorSpec(type='int', shape=()).signature(batched=True),
                current_x=self.output_spec().signature(batched=True),
                current_internals=self.internals_spec.signature(batched=True)
            )

        elif function == 'past_horizon':
            return SignatureDict(
                singleton=TensorSpec(type='int', shape=()).signature(batched=False)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=0)
    def past_horizon(self, *, on_policy):
        if self.temporal_processing == 'iterative' and on_policy:
            return tf_util.constant(value=0, dtype='int')
        else:
            return self.horizon.value()

    @tf_function(num_args=3, overwrites_signature=True)
    def apply(self, *, x, horizons, internals):
        zero = tf_util.constant(value=0, dtype='int')
        one = tf_util.constant(value=1, dtype='int')

        batch_size = tf_util.cast(x=tf.shape(input=horizons)[0], dtype='int')

        zeros = tf_util.zeros(shape=(batch_size,), dtype='int')
        ones = tf_util.ones(shape=(batch_size,), dtype='int')

        # including 0th step
        horizon = self.horizon.value() + one
        # in case of longer horizon than necessary (e.g. main vs baseline policy)
        starts = horizons[:, 0] + tf.maximum(x=(horizons[:, 1] - horizon), y=zeros)
        lengths = horizons[:, 1] - tf.maximum(x=(horizons[:, 1] - horizon), y=zeros)
        horizon = tf.minimum(x=horizon, y=tf.math.reduce_max(input_tensor=lengths, axis=0))
        output_spec = self.output_spec()

        if self.temporal_processing == 'cumulative':
            if self.horizon.is_constant(value=0):
                x = self.iterative_apply(xs=x, lengths=ones)

            else:
                def body(x, indices, remaining, xs):
                    current_x = tf.gather(params=x, indices=indices)
                    current_x = tf.expand_dims(input=current_x, axis=1)
                    xs = tf.concat(values=(xs, current_x), axis=1)
                    remaining -= tf.where(
                        condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones
                    )
                    indices += tf.where(condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones)
                    return x, indices, remaining, xs

                initial_xs = tf_util.zeros(
                    shape=((batch_size, 0) + output_spec.shape), dtype=output_spec.type
                )

                _, final_indices, final_remaining, xs = tf.while_loop(
                    cond=tf_util.always_true, body=body, loop_vars=(x, starts, lengths, initial_xs),
                    maximum_iterations=tf_util.int64(x=horizon)
                )

                x = self.cumulative_apply(xs=xs, lengths=lengths)

        elif self.temporal_processing == 'iterative':
            if self.horizon.is_constant(value=0):
                x, final_internals = self.iterative_apply(x=x, internals=internals)

            else:
                initial_x = tf_util.zeros(
                    shape=((batch_size,) + output_spec.shape), dtype=output_spec.type
                )

                signature = self.input_signature(function='iterative_body')
                internals = signature['current_internals'].kwargs_to_args(kwargs=internals)
                _, final_indices, final_remaining, x, final_internals = tf.while_loop(
                    cond=tf_util.always_true, body=self.iterative_body,
                    loop_vars=(x, starts, lengths, initial_x, internals),
                    maximum_iterations=tf_util.int32(x=horizon)
                )
                internals = signature['current_internals'].args_to_kwargs(args=final_internals)

        assertions = list()
        if self.config.create_tf_assertions:
            assertions.append(tf.debugging.assert_equal(
                x=final_indices, y=(tf.math.cumsum(x=lengths) - ones)
            ))
            assertions.append(tf.debugging.assert_equal(
                x=tf.math.reduce_sum(input_tensor=final_remaining), y=zero
            ))

        with tf.control_dependencies(control_inputs=assertions):
            if self.temporal_processing == 'cumulative':
                return tf_util.identity(input=super().apply(x=x))
            elif self.temporal_processing == 'iterative':
                return tf_util.identity(input=super().apply(x=x)), internals

    @tf_function(num_args=5, is_loop_body=True)
    def iterative_body(self, x, indices, remaining, current_x, current_internals):
        batch_size = tf_util.cast(x=tf.shape(input=current_x)[0], dtype='int')
        zeros = tf_util.zeros(shape=(batch_size,), dtype='int')
        ones = tf_util.ones(shape=(batch_size,), dtype='int')

        current_x = tf.gather(params=x, indices=indices)
        next_x, next_internals = self.iterative_apply(
            x=current_x, internals=current_internals
        )

        with tf.control_dependencies(control_inputs=(current_x, next_x)):
            is_finished = tf.math.equal(x=remaining, y=zeros)
            if isinstance(next_internals, dict):
                for name, current_internal, next_internal in current_internals.zip_items(
                    next_internals
                ):
                    condition = is_finished
                    for _ in range(tf_util.rank(x=current_internal) - 1):
                        condition = tf.expand_dims(input=condition, axis=1)
                    next_internals[name] = tf.where(
                        condition=condition, x=current_internal, y=next_internal
                    )

            else:
                condition = is_finished
                for _ in range(tf_util.rank(x=current_internals) - 1):
                    condition = tf.expand_dims(input=condition, axis=1)
                next_internals = tf.where(
                    condition=condition, x=current_internals, y=next_internals
                )

            remaining -= tf.where(condition=is_finished, x=zeros, y=ones)
            indices += tf.where(
                condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones
            )

        return x, indices, remaining, next_x, next_internals

    @tf_function(num_args=1)
    def cumulative_apply(self, *, xs, lengths):
        raise NotImplementedError

    @tf_function(num_args=2)
    def iterative_apply(self, *, x, internals):
        raise NotImplementedError


class TransformationBase(Layer):
    """
    Base class for transformation layers.

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: false).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: none).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
        kwargs: Additional arguments for potential parent class.
    """

    def __init__(
        self, *, size, bias=False, activation=None, dropout=0.0, vars_trainable=True,
        l2_regularization=None, name=None, input_spec=None, **kwargs
    ):
        super().__init__(
            l2_regularization=l2_regularization, name=name, input_spec=input_spec, **kwargs
        )

        self.squeeze = (size == 0)
        self.size = max(size, 1)
        self.bias = bias

        if activation is None:
            self.activation = None
        else:
            self.activation = self.submodule(
                name='activation', module='activation', modules=tensorforce.core.layer_modules,
                nonlinearity=activation, input_spec=self.output_spec()
            )

        if dropout == 0.0:
            self.dropout = None
        else:
            self.dropout = self.submodule(
                name='dropout', module='dropout', modules=tensorforce.core.layer_modules,
                rate=dropout, input_spec=self.output_spec()
            )

        self.vars_trainable = vars_trainable

    def initialize(self):
        super().initialize()

        if isinstance(self.bias, str):
            # Hack for Rnn to avoid name clash with Keras variable name
            self.bias = self.variable(
                name=self.bias, spec=TensorSpec(type='float', shape=(self.size,)),
                initializer='zeros', is_trainable=self.vars_trainable, is_saved=True
            )
        elif self.bias:
            self.bias = self.variable(
                name='bias', spec=TensorSpec(type='float', shape=(self.size,)), initializer='zeros',
                is_trainable=self.vars_trainable, is_saved=True
            )
        else:
            self.bias = None

    @tf_function(num_args=1)
    def apply(self, *, x):
        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.squeeze:
            x = tf.squeeze(input=x, axis=-1)

        if self.activation is not None:
            x = self.activation.apply(x=x)

        if self.dropout is not None:
            x = self.dropout.apply(x=x)

        return x
