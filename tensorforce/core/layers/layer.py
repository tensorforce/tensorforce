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

from tensorforce import TensorforceError, util
import tensorforce.core
from tensorforce.core import Module, parameter_modules, tf_function
from tensorforce.core.parameters import Parameter


class Layer(Module):
    """
    Base class for neural network layers.

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    # registered_layers = OrderedDict()

    def __init__(self, name, input_spec=None, summary_labels=None, l2_regularization=None):
        super().__init__(
            name=name, summary_labels=summary_labels, l2_regularization=l2_regularization
        )

        module = self
        while isinstance(module, Layer):
            module = module.parent
        if isinstance(module, (tensorforce.core.networks.LayerbasedNetwork)):
            module.registered_layers[self.name] = self

        self.input_spec = util.valid_value_spec(
            value_spec=self.default_input_spec(), accept_underspecified=True, return_normalized=True
        )

        if input_spec is not None:
            input_spec = util.valid_value_spec(
                value_spec=input_spec, accept_underspecified=True, return_normalized=True
            )

            self.input_spec = util.unify_value_specs(
                value_spec1=self.input_spec, value_spec2=input_spec
            )

    def default_input_spec(self):
        """
        Returns the general, context-independent input tensor specification of this layer.

        Returns:
            General input tensor specification.
        """
        return dict(type=None, shape=None)

    def output_spec(self):
        """
        Returns the output tensor specification.

        Returns:
            Output tensor specification.
        """
        return dict(self.input_spec)

    def add_module(self, *args, **kwargs):
        layer = super().add_module(*args, **kwargs)

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(name='layer', argument='sub-module', dtype=type(layer))

        return layer

    def input_signature(self, function):
        if function == 'apply':
            return [util.to_tensor_spec(value_spec=self.input_spec, batched=True)]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=1)
    def apply(self, x):
        return x

class Register(Layer):
    """
    Tensor retrieval layer, which is useful when defining more complex network architectures which
    do not follow the sequential layer-stack pattern, for instance, when handling multiple inputs
    (specification key: `register`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        tensor (string): Name under which tensor will be registered
            (<span style="color:#C00000"><b>required</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, tensor, input_spec=None, summary_labels=None):
        """
        Register layer constructor.

        Args:
        """
        super().__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

        if not isinstance(tensor, str):
            raise TensorforceError.type(name='register', argument='tensor', dtype=type(tensor))

        self.tensor = tensor


class Retrieve(Layer):
    """
    Tensor retrieval layer, which is useful when defining more complex network architectures which
    do not follow the sequential layer-stack pattern, for instance, when handling multiple inputs
    (specification key: `retrieve`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        tensors (iter[string]): Names of global tensors to retrieve, for instance, state names or
            previously registered global tensor names
            (<span style="color:#C00000"><b>required</b></span>).
        aggregation ('concat' | 'product' | 'stack' | 'sum'): Aggregation type in case of multiple
            tensors
            (<span style="color:#00C000"><b>default</b></span>: 'concat').
        axis (int >= 0): Aggregation axis, excluding batch axis
            (<span style="color:#00C000"><b>default</b></span>: 0).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, tensors, aggregation='concat', axis=0, input_spec=None, summary_labels=None
    ):
        super().__init__(name=name, input_spec=None, summary_labels=summary_labels)

        self.input_spec = util.valid_values_spec(
            values_spec=input_spec, accept_underspecified=True, return_normalized=True
        )

        if not util.is_iterable(x=tensors):
            raise TensorforceError.type(name='retrieve', argument='tensors', dtype=type(tensors))
        elif util.is_iterable(x=tensors):
            if len(tensors) == 0:
                raise TensorforceError.value(
                    name='retrieve', argument='tensors', value=tensors, hint='zero length'
                )
            elif len(tensors) != len(input_spec):
                raise TensorforceError.value(
                    name='retrieve', argument='tensors', value=tensors,
                    condition=('input-spec length ' + str(len(input_spec)))
                )
        if aggregation not in ('concat', 'product', 'stack', 'sum'):
            raise TensorforceError.value(
                name='retrieve', argument='aggregation', value=aggregation,
                hint='not in {concat,product,stack,sum}'
            )

        self.tensors = tuple(tensors)
        self.aggregation = aggregation
        self.axis = axis

    def output_spec(self):
        output_spec = super().output_spec()

        if len(self.tensors) == 1:
            return output_spec[self.tensors[0]]

        # Get tensor types and shapes
        dtypes = list()
        shapes = list()
        for spec in output_spec.values():
            dtypes.append(spec['type'])
            shapes.append(spec['shape'])

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

        # Missing num_values, min/max_value!!!
        return dict(type=dtype, shape=shape)

    @tf_function(num_args=1)
    def apply(self, x):
        if len(self.tensors) == 1:
            return x[self.tensors[0]]

        tensors = list(x.values())

        shape = self.output_spec()['shape']
        for n, tensor in enumerate(tensors):
            for axis in range(util.rank(x=tensor), len(shape)):
                tensor = tf.expand_dims(input=tensor, axis=axis)
            tensors[n] = tensor

        if self.aggregation == 'concat':
            x = tf.concat(values=tensors, axis=(self.axis + 1))

        elif self.aggregation == 'product':
            x = tf.stack(values=tensors, axis=(self.axis + 1))
            x = tf.reduce_prod(input_tensor=x, axis=(self.axis + 1))

        elif self.aggregation == 'stack':
            x = tf.stack(values=tensors, axis=(self.axis + 1))

        elif self.aggregation == 'sum':
            x = tf.stack(values=tensors, axis=(self.axis + 1))
            x = tf.reduce_sum(input_tensor=x, axis=(self.axis + 1))

        return x


class TransformationBase(Layer):
    """
    Base class for transformation layers.

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
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
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments for potential parent class.
    """

    def __init__(
        self, name, size, bias=False, activation=None, dropout=0.0, vars_trainable=True,
        input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, **kwargs
        )

        self.squeeze = (size == 0)
        self.size = max(size, 1)
        self.bias = bias
        if activation is None:
            self.activation = None
        else:
            self.activation = self.add_module(
                name='activation', module='activation', modules=tensorforce.core.layer_modules,
                nonlinearity=activation, input_spec=self.output_spec()
            )
        if dropout is None or dropout == 0.0:
            self.dropout = None
        else:
            self.dropout = self.add_module(
                name='dropout', module='dropout', modules=tensorforce.core.layer_modules,
                rate=dropout, input_spec=self.output_spec()
            )
        self.vars_trainable = vars_trainable

    def tf_initialize(self):
        super().tf_initialize()

        if self.bias:
            self.bias = self.add_variable(
                name='bias', dtype='float', shape=(self.size,), is_trainable=self.vars_trainable,
                initializer=('zeros' if self.vars_trainable else 'normal')
            )

        else:
            self.bias = None

    @tf_function(num_args=1)
    def apply(self, x):
        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.squeeze:
            x = tf.squeeze(input=x, axis=-1)

        if self.activation is not None:
            x = self.activation.apply(x=x)

        if self.dropout is not None:
            x = self.dropout.apply(x=x)

        return x


class TemporalLayer(Layer):
    """
    Base class for temporal layers, i.e. layers whose output depends on previous states.

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        temporal_processing ('cumulative' | 'iterative'): Temporal processing type
            (<span style="color:#C00000"><b>required</b></span>).
        horizon (parameter, long >= 0): Past horizon
            (<span style="color:#C00000"><b>required</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments for potential parent class.
    """

    def __init__(
        self, name, temporal_processing, horizon, input_spec=None, summary_labels=None,
        l2_regularization=None, **kwargs
    ):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization, **kwargs
        )

        if temporal_processing not in ('cumulative', 'iterative'):
            raise TensorforceError.value(
                name='temporal-layer', argument='temporal_processing', value=temporal_processing,
                hint='not in {cumulative,iterative}'
            )

        self.temporal_processing = temporal_processing

        if self.temporal_processing == 'cumulative':
            assert len(self.__class__.internals_spec(layer=self)) == 0

        self.horizon = self.add_module(
            name='horizon', module=horizon, modules=parameter_modules, is_trainable=False,
            dtype='long', min_value=0
        )

    @classmethod
    def internals_spec(cls, layer=None, **kwargs):
        return OrderedDict()

    def internals_init(self):
        return OrderedDict()

    def max_past_horizon(self, on_policy):
        if self.temporal_processing == 'iterative' and on_policy:
            return 0
        else:
            return self.horizon.max_value()

    def input_signature(self, function):
        if function == 'apply':
            return [
                util.to_tensor_spec(value_spec=self.input_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(
                    value_spec=self.__class__.internals_spec(layer=self), batched=True
                )
            ]

        elif function == 'cumulative_apply':
            value_spec = dict(self.input_spec)
            value_spec['shape'] = (None,) + value_spec['shape']
            return [
                util.to_tensor_spec(value_spec=value_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=True)
            ]

        elif function == 'iterative_step':
            return [
                util.to_tensor_spec(value_spec=self.input_spec, batched=True),
                util.to_tensor_spec(
                    value_spec=self.__class__.internals_spec(layer=self), batched=True
                )
            ]

        elif function == 'past_horizon':
            return ()

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=0)
    def past_horizon(self, on_policy):
        if self.temporal_processing == 'iterative' and on_policy:
            return tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        else:
            return self.horizon.value()

    @tf_function(num_args=3)
    def apply(self, x, horizons, internals):
        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))

        if util.tf_dtype(dtype='long') in (tf.int32, tf.int64):
            batch_size = tf.shape(input=horizons, out_type=util.tf_dtype(dtype='long'))[0]
        else:
            batch_size = tf.dtypes.cast(
                x=tf.shape(input=horizons)[0], dtype=util.tf_dtype(dtype='long')
            )

        zeros = tf.zeros(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
        ones = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))

        # including 0th step
        horizon = self.horizon.value() + one
        # in case of longer horizon than necessary (e.g. main vs baseline policy)
        starts = horizons[:, 0] + tf.maximum(x=(horizons[:, 1] - horizon), y=zeros)
        lengths = horizons[:, 1] - tf.maximum(x=(horizons[:, 1] - horizon), y=zeros)
        horizon = tf.minimum(x=horizon, y=tf.math.reduce_max(input_tensor=lengths, axis=0))
        output_spec = self.output_spec()

        if self.temporal_processing == 'cumulative':

            def body(indices, remaining, xs):
                current_x = tf.gather(params=x, indices=indices)
                current_x = tf.expand_dims(input=current_x, axis=1)
                xs = tf.concat(values=(xs, current_x), axis=1)
                remaining -= tf.where(
                    condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones
                )
                indices += tf.where(condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones)
                return indices, remaining, xs

            initial_xs = tf.zeros(
                shape=((batch_size, 0) + output_spec['shape']),
                dtype=util.tf_dtype(dtype=output_spec['type'])
            )

            final_indices, final_remaining, xs = self.while_loop(
                cond=util.tf_always_true, body=body, loop_vars=(starts, lengths, initial_xs),
                back_prop=True, maximum_iterations=horizon
            )

            x = self.cumulative_apply(xs=xs, lengths=lengths)

        elif self.temporal_processing == 'iterative':

            def body(indices, remaining, current_x, current_internals):
                current_x = tf.gather(params=x, indices=indices)
                next_x, next_internals = self.iterative_step(
                    x=current_x, internals=current_internals
                )

                with tf.control_dependencies(control_inputs=(current_x, next_x)):
                    is_finished = tf.math.equal(x=remaining, y=zeros)
                    if isinstance(next_internals, dict):
                        for name, current_internal, next_internal in util.zip_items(
                            current_internals, next_internals
                        ):
                            condition = is_finished
                            for _ in range(util.rank(x=current_internal) - 1):
                                condition = tf.expand_dims(input=condition, axis=1)
                            next_internals[name] = tf.where(
                                condition=condition, x=current_internal, y=next_internal
                            )

                    else:
                        condition = is_finished
                        for _ in range(util.rank(x=current_internals) - 1):
                            condition = tf.expand_dims(input=condition, axis=1)
                        next_internals = tf.where(
                            condition=condition, x=current_internals, y=next_internals
                        )

                    remaining -= tf.where(condition=is_finished, x=zeros, y=ones)
                    indices += tf.where(
                        condition=tf.math.equal(x=remaining, y=zeros), x=zeros, y=ones
                    )

                return indices, remaining, next_x, next_internals

            initial_x = tf.zeros(
                shape=((batch_size,) + output_spec['shape']),
                dtype=util.tf_dtype(dtype=output_spec['type'])
            )

            final_indices, final_remaining, x, internals = self.while_loop(
                cond=util.tf_always_true, body=body,
                loop_vars=(starts, lengths, initial_x, internals), back_prop=True,
                maximum_iterations=horizon
            )

        assertions = [
            tf.debugging.assert_equal(x=final_indices, y=(tf.math.cumsum(x=lengths) - ones)),
            tf.debugging.assert_equal(x=tf.math.reduce_sum(input_tensor=final_remaining), y=zero)
        ]

        with tf.control_dependencies(control_inputs=assertions):
            if self.temporal_processing == 'cumulative':
                return util.identity_operation(x=super().apply(x=x))
            elif self.temporal_processing == 'iterative':
                return util.identity_operation(x=super().apply(x=x)), internals

    @tf_function(num_args=1)
    def cumulative_apply(self, xs, lengths):
        raise NotImplementedError

    @tf_function(num_args=2)
    def iterative_step(self, x, internals):
        raise NotImplementedError
