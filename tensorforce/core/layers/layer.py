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
from tensorforce.core import Module


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

        if not self.input_spec['batched']:
            raise TensorforceError.value(
                name='default input-spec', argument='batched', value=self.input_spec['batched']
            )

        if input_spec is not None:
            input_spec = util.valid_value_spec(
                value_spec=input_spec, accept_underspecified=True, return_normalized=True
            )

            if not input_spec['batched']:
                raise TensorforceError.value(
                    name='input-spec', argument='batched', value=input_spec['batched']
                )

            self.input_spec = util.unify_value_specs(
                value_spec1=self.input_spec, value_spec2=input_spec
            )

        # Copy so that spec can be modified
        self.output_spec = self.get_output_spec(input_spec=dict(self.input_spec))
        self.output_spec = util.valid_value_spec(
            value_spec=self.output_spec, accept_underspecified=True, return_normalized=True
        )

        if not self.output_spec['batched']:
            raise TensorforceError.value(
                name='output_spec', argument='batched', value=self.output_spec['batched']
            )

    def default_input_spec(self):
        raise NotImplementedError

    def get_output_spec(self, input_spec):
        return input_spec

    def internals_spec(self):
        """
        Returns the internal states specification.

        Returns:
            Internal states specification
        """
        specification = OrderedDict()

        for layer in self.modules.values():
            for name, spec in layer.internals_spec().items():
                name = layer.name + '-' + name
                if name in specification:
                    raise TensorforceError.unexpected()
                # check valid names!!!
                specification[name] = spec

        return specification

    def internals_init(self):
        initialization = OrderedDict()

        for layer in self.modules.values():
            for name, init in layer.internals_init().items():
                # check valid names!!!
                initialization[layer.name + '-' + name] = init

        return initialization

    def add_module(self, *args, **kwargs):
        layer = super().add_module(*args, **kwargs)

        if not isinstance(layer, Layer):
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
        if name[-6:] != '.apply':
            return super().create_tf_function(name=name, tf_function=tf_function)

        def validated_tf_function(x, **internals):
            if not util.is_consistent_with_value_spec(value_spec=self.input_spec, x=x):
                raise TensorforceError("Invalid input arguments for tf_apply.")
            if not all(
                util.is_consistent_with_value_spec(value_spec=spec, x=internals[name])
                for name, spec in self.internals_spec().items()
            ):
                raise TensorforceError("Invalid input arguments for tf_apply.")

            if len(internals) > 0:
                x, internals = tf_function(x=x, **internals)
            else:
                x = tf_function(x=x)

            if not util.is_consistent_with_value_spec(value_spec=self.output_spec, x=x):
                raise TensorforceError("Invalid output arguments for tf_apply.")
            if not all(
                util.is_consistent_with_value_spec(value_spec=spec, x=internals[name])
                for name, spec in self.internals_spec().items()
            ):
                raise TensorforceError("Invalid input arguments for tf_apply.")

            if len(internals) > 0:
                return x, internals
            else:
                return x

        return super().create_tf_function(name=name, tf_function=validated_tf_function)


class Retrieve(Layer):
    """
    Retrieve layer.
    """

    def __init__(self, name, tensors, aggregation='concat', axis=1, input_spec=None):
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

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        if len(self.tensors) == 1:
            if self.tensors[0] in Module.global_tensors_spec:
                return Module.global_tensors_spec[self.tensors[0]]
            else:
                raise TensorforceError.value(
                    name='retrieve-layer', argument='tensors', value=self.tensors
                )

        # Get tensor types and shapes
        dtypes = list()
        shapes = list()
        for tensor in self.tensors:
            # Tensor specification
            if tensor == '*':
                spec = input_spec
            elif tensor in Module.global_tensors_spec:
                spec = Module.global_tensors_spec[tensor]
            else:
                raise TensorforceError.value(
                    name='retrieve-layer', argument='tensors', value=self.tensors
                )
            dtypes.append(spec['type'])
            shapes.append(spec['shape'])

        # Check tensor types
        if all(dtype == dtypes[0] for dtype in dtypes):
            dtype = dtypes[0]
        else:
            raise TensorforceError.value(name='tensor types', value=dtypes)

        # Check and unify tensor shapes
        max_shape = ()
        for shape in shapes:
            if any(x != y for x, y in zip(shape, max_shape)):
                pass
                # raise TensorforceError.value(name='tensor shapes', value=shapes)
            elif len(shape) > len(max_shape):
                max_shape = shape

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
            for rank in range(util.rank(x=tensor), len(shape)):
                tensor = tf.expand_dims(input=tensor, axis=rank)
            tensors[n] = tensor

        if self.aggregation == 'concat':
            x = tf.concat(values=tensors, axis=self.axis)

        elif self.aggregation == 'product':
            x = tf.stack(values=tensors, axis=self.axis)
            x = tf.reduce_prod(input_tensor=x, axis=self.axis)

        elif self.aggregation == 'stack':
            x = tf.stack(values=tensors, axis=self.axis)

        elif self.aggregation == 'sum':
            x = tf.stack(values=tensors, axis=self.axis)
            x = tf.reduce_sum(input_tensor=x, axis=self.axis)

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

        input_spec = self.output_spec
        if activation is None:
            self.activation = None
        else:
            self.activation = self.add_module(
                name='activation', module='activation', modules=layer_modules,
                nonlinearity=activation, input_spec=input_spec
            )
            input_spec = self.activation.output_spec

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = self.add_module(
                name='dropout', module='dropout', modules=layer_modules, rate=dropout,
                input_spec=input_spec
            )

    def specify_input_output_spec(self, input_spec):
        super().specify_input_output_spec(input_spec=input_spec)

        input_spec = self.output_spec
        if self.activation is not None:
            self.activation.specify_input_output_spec(input_spec=input_spec)
            input_spec = self.activation.output_spec
        if self.dropout is not None:
            self.dropout.specify_input_output_spec(input_spec=input_spec)

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
