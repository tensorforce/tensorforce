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

from tensorforce import TensorforceError, util
from tensorforce.core import Module
from tensorforce.core.parameters import Parameter


class Layer(Module):
    """
    Neural network layer base class.
    """

    def __init__(self, name, input_spec=None, l2_regularization=None, summary_labels=None):
        """
        Layer constructor.

        Args:
            input_spec (specification): Input tensor specification.
        """
        super().__init__(
            name=name, l2_regularization=l2_regularization, summary_labels=summary_labels
        )

        self.input_spec = self.default_input_spec()
        self.input_spec = util.valid_value_spec(
            value_spec=self.input_spec, accept_underspecified=True, return_normalized=True
        )

        if input_spec is not None:
            input_spec = util.valid_value_spec(
                value_spec=input_spec, accept_underspecified=True, return_normalized=True
            )

            self.input_spec = util.unify_value_specs(
                value_spec1=self.input_spec, value_spec2=input_spec
            )

        # Copy so that spec can be modified
        self.output_spec = self.get_output_spec(input_spec=dict(self.input_spec))
        self.output_spec = util.valid_value_spec(
            value_spec=self.output_spec, accept_underspecified=True, return_normalized=True
        )

    def default_input_spec(self):
        raise NotImplementedError

    def get_output_spec(self, input_spec):
        return input_spec

    def add_module(self, *args, **kwargs):
        layer = super().add_module(*args, **kwargs)

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(name='layer', argument='sub-module', value=layer)

        return layer

    def tf_apply(self, x):
        """
        Creates the TensorFlow operations for applying the layer to the given input.

        Args:
            x: Layer input tensor.

        Returns:
            Layer output tensor.
        """
        raise NotImplementedError

    def create_tf_function(self, name, tf_function):
        # if name[-6:] != '.apply':
        if tf_function.__name__ != 'tf_apply':
            return super().create_tf_function(name=name, tf_function=tf_function)

        def validated_tf_function(x):
            if self.input_spec is not None and \
                    not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
                raise TensorforceError("Invalid input arguments for tf_apply.")

            x = tf_function(x=x)

            if self.output_spec is not None and \
                    not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
                raise TensorforceError("Invalid output arguments for tf_apply.")

            return x

        return super().create_tf_function(name=name, tf_function=validated_tf_function)


class Retrieve(Layer):
    """
    Retrieve layer.
    """

    def __init__(self, name, tensors, aggregation='concat', axis=0, input_spec=None):
        """
        Retrieve constructor.

        Args:
            tensors (iter[string]): Global names of tensors to retrieve.
            aggregation ('concat' | 'product' | 'stack' | 'sum'): Aggregation type.
            axis (int >= 0): Aggregation axis (excluding batch axis).

        """
        if not isinstance(tensors, str) and not util.is_iterable(x=tensors):
            raise TensorforceError.type(name='retrieve', argument='tensors', value=tensors)
        elif util.is_iterable(x=tensors) and len(tensors) == 0:
            raise TensorforceError.value(name='retrieve', argument='tensors', value=tensors)
        if aggregation not in ('concat', 'product', 'stack', 'sum'):
            raise TensorforceError.value(
                name='retrieve', argument='aggregation', value=aggregation
            )

        self.tensors = (tensors,) if isinstance(tensors, str) else tuple(tensors)
        self.aggregation = aggregation
        self.axis = axis

        super().__init__(name=name, input_spec=input_spec, l2_regularization=0.0)

        self.input_spec = None

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        if len(self.tensors) == 1:
            return Module.get_tensor_spec(name=self.tensors[0])

        # Get tensor types and shapes
        dtypes = list()
        shapes = list()
        for tensor in self.tensors:
            # Tensor specification
            if tensor == '*':
                spec = input_spec
            else:
                spec = Module.get_tensor_spec(name=tensor)
            dtypes.append(spec['type'])
            shapes.append(spec['shape'])

        # Check tensor types
        if all(dtype == dtypes[0] for dtype in dtypes):
            dtype = dtypes[0]
        else:
            raise TensorforceError.value(name='tensor types', value=dtypes)

        if self.aggregation == 'concat':
            if any(len(shape) != len(shapes[0]) for shape in shapes):
                raise TensorforceError.value(name='tensor shapes', value=shapes)
            elif any(
                shape[n] != shapes[0][n] for shape in shapes for n in range(len(shape))
                if n != self.axis
            ):
                raise TensorforceError.value(name='tensor shapes', value=shapes)
            shape = tuple(
                sum(shape[n] for shape in shapes) if n == self.axis else shapes[0][n]
                for n in range(len(shapes[0]))
            )

        elif self.aggregation == 'stack':
            if any(len(shape) != len(shapes[0]) for shape in shapes):
                raise TensorforceError.value(name='tensor shapes', value=shapes)
            elif any(shape[n] != shapes[0][n] for shape in shapes for n in range(len(shape))):
                raise TensorforceError.value(name='tensor shapes', value=shapes)
            shape = tuple(
                len(shapes) if n == self.axis else shapes[0][n - int(n > self.axis)]
                for n in range(len(shapes[0]) + 1)
            )

        else:
            # Check and unify tensor shapes
            for shape in shapes:
                if len(shape) != len(shapes[0]):
                    raise TensorforceError.value(name='tensor shapes', value=shapes)
                if any(x != y and x != 1 and y != 1 for x, y in zip(shape, shapes[0])):
                    raise TensorforceError.value(name='tensor shapes', value=shapes)
            shape = tuple(max(shape[n] for shape in shapes) for n in range(len(shapes[0])))

        # Missing num_values, min/max_value!!!
        return dict(type=dtype, shape=shape)

    def tf_apply(self, x):
        if len(self.tensors) == 1:
            if self.tensors == '*':
                return x
            else:
                return Module.retrieve_tensor(name=self.tensors[0])

        tensors = list()
        for tensor in self.tensors:
            if tensor == '*':
                tensors.append(x)
            else:
                tensors.append(Module.retrieve_tensor(name=tensor))

        shape = self.output_spec['shape']
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


class Register(Layer):
    """
    Register layer.
    """

    def __init__(self, name, tensor, input_spec=None):
        """
        Register constructor.

        Args:
            tensor (string): Global name for registered tensor.
        """
        if not isinstance(tensor, str):
            raise TensorforceError.type(name='register', argument='tensor', value=tensor)

        self.tensor = tensor

        super().__init__(name=name, input_spec=input_spec, l2_regularization=0.0)

        Module.register_tensor(name=self.tensor, spec=self.input_spec, batched=True)

        self.output_spec = None

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def tf_apply(self, x):
        Module.update_tensor(name=self.tensor, tensor=x)

        return x


class TransformationBase(Layer):
    """
    Transformation layer base class.
    """

    def __init__(
        self, name, size, bias=False, activation=None, dropout=None, input_spec=None,
        l2_regularization=None, summary_labels=None
    ):
        """
        Transformation constructor.

        Args:
            size (int >= 0): Layer size.
            bias (bool): ???
            activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
                'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity.
            dropout (0.0 <= float < 1.0): Dropout rate.
        """
        self.squeeze = (size == 0)
        self.size = max(size, 1)
        self.bias = bias
        self.activation = None
        self.dropout = None

        super().__init__(
            name=name, input_spec=input_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

        from tensorforce.core import layer_modules

        if activation is None:
            self.activation = None
        else:
            self.activation = self.add_module(
                name='activation', module='activation', modules=layer_modules,
                nonlinearity=activation, input_spec=self.output_spec
            )

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = self.add_module(
                name='dropout', module='dropout', modules=layer_modules, rate=dropout,
                input_spec=self.output_spec
            )

    def specify_input_output_spec(self, input_spec):
        super().specify_input_output_spec(input_spec=input_spec)

        if self.activation is not None:
            self.activation.specify_input_output_spec(input_spec=self.output_spec)
            assert self.activation.output_spec == self.output_spec
        if self.dropout is not None:
            self.dropout.specify_input_output_spec(input_spec=self.output_spec)
            assert self.dropout.output_spec == self.output_spec

    def tf_initialize(self):
        super().tf_initialize()

        if self.bias:
            self.bias = self.add_variable(
                name='bias', dtype='float', shape=(self.size,), is_trainable=True,
                initializer='zeros'
            )

        else:
            self.bias = None

    def tf_apply(self, x):
        # shape = self.get_output_spec()['shape']
        # if self.squeeze:
        #     shape = shape + (1,)
        # if util.dtype(x=x) != 'float' or util.shape(x=x)[1:] != shape:
        #     raise TensorforceError("Invalid input tensor for generic layer: {}.".format(x))

        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.squeeze:
            x = tf.squeeze(input=x, axis=-1)

        if self.activation is not None:
            x = self.activation.apply(x=x)

        if self.dropout is not None:
            x = self.dropout.apply(x=x)

        return x
