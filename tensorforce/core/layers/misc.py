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

from collections import Counter

import tensorflow as tf

from tensorforce import TensorforceError, util
import tensorforce.core
from tensorforce.core import Module, parameter_modules
from tensorforce.core.layers import Layer


class Activation(Layer):
    """
    Activation layer (specification key: `activation`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        nonlinearity ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Nonlinearity
            (<span style="color:#C00000"><b>required</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, nonlinearity, input_spec=None, summary_labels=None
    ):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

        # Nonlinearity
        if nonlinearity not in (
            'crelu', 'elu', 'leaky-relu', 'none', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus',
            'softsign', 'swish', 'tanh'
        ):
            raise TensorforceError('Invalid nonlinearity: {}'.format(self.nonlinearity))
        self.nonlinearity = nonlinearity

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def tf_apply(self, x):
        if self.nonlinearity == 'crelu':
            x = tf.nn.crelu(features=x)

        elif self.nonlinearity == 'elu':
            x = tf.nn.elu(features=x)

        elif self.nonlinearity == 'leaky-relu':
            x = tf.nn.leaky_relu(features=x, alpha=0.2)  # alpha argument???

        elif self.nonlinearity == 'none':
            pass

        elif self.nonlinearity == 'relu':
            x = tf.nn.relu(features=x)
            x = self.add_summary(
                label='relu', name='relu', tensor=tf.math.zero_fraction(value=x), pass_tensors=x
            )

        elif self.nonlinearity == 'selu':
            x = tf.nn.selu(features=x)

        elif self.nonlinearity == 'sigmoid':
            x = tf.sigmoid(x=x)

        elif self.nonlinearity == 'softmax':
            x = tf.nn.softmax(logits=x)

        elif self.nonlinearity == 'softplus':
            x = tf.nn.softplus(features=x)

        elif self.nonlinearity == 'softsign':
            x = tf.nn.softsign(features=x)

        elif self.nonlinearity == 'swish':
            # https://arxiv.org/abs/1710.05941
            x = tf.sigmoid(x=x) * x

        elif self.nonlinearity == 'tanh':
            x = tf.nn.tanh(x=x)

        return x


class Block(Layer):
    """
    Block of layers (specification key: `block`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        layers (iter[specification]): Layers configuration, see [layers](../modules/layers.html)
            (<span style="color:#C00000"><b>required</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
    """

    def __init__(self, name, layers, input_spec=None):
        # TODO: handle internal states and combine with layered network
        if len(layers) == 0:
            raise TensorforceError.unexpected()

        self._input_spec = input_spec
        self.layers = layers

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=None, l2_regularization=0.0
        )

    def default_input_spec(self):
        layer_counter = Counter()
        for n, layer_spec in enumerate(self.layers):
            if 'name' in layer_spec:
                layer_spec = dict(layer_spec)
                layer_name = layer_spec.pop('name')
            else:
                if isinstance(layer_spec.get('type'), str):
                    layer_type = layer_spec['type']
                else:
                    layer_type = 'layer'
                layer_name = layer_type + str(layer_counter[layer_type])
                layer_counter[layer_type] += 1

            # layer_name = self.name + '-' + layer_name
            self.layers[n] = self.add_module(
                name=layer_name, module=layer_spec, modules=tensorforce.core.layer_modules,
                input_spec=self._input_spec
            )
            self._input_spec = self.layers[n].output_spec

        return self.layers[0].default_input_spec()

    def get_output_spec(self, input_spec):
        for layer in self.layers:
            input_spec = layer.get_output_spec(input_spec=input_spec)
        return input_spec

    def tf_apply(self, x):
        for layer in self.layers:
            x = layer.apply(x=x)
        return x


class Dropout(Layer):
    """
    Dropout layer (specification key: `dropout`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        rate (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#C00000"><b>required</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, rate, input_spec=None, summary_labels=None):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

        # Rate
        self.rate = self.add_module(
            name='rate', module=rate, modules=parameter_modules, dtype='float'
        )

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def set_input_spec(self, spec):
        super().set_input_spec(spec=spec)

        if spec['type'] != 'float':
            raise TensorforceError(
                "Invalid input type for dropout layer: {}.".format(spec['type'])
            )

    def tf_apply(self, x):
        rate = self.rate.value()

        def no_dropout():
            return x

        def apply_dropout():
            dropout = tf.nn.dropout(x=x, rate=rate)
            return self.add_summary(
                label='dropout', name='dropout', tensor=tf.math.zero_fraction(value=dropout),
                pass_tensors=dropout
            )

        skip_dropout = tf.math.logical_not(x=Module.retrieve_tensor(name='optimization'))
        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        skip_dropout = tf.math.logical_or(x=skip_dropout, y=tf.math.equal(x=rate, y=zero))
        return self.cond(pred=skip_dropout, true_fn=no_dropout, false_fn=apply_dropout)


class Function(Layer):
    """
    Custom TensorFlow function layer (specification key: `function`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        function (lambda[x -> x]): TensorFlow function
            (<span style="color:#C00000"><b>required</b></span>).
        output_spec (specification): Output tensor specification containing type and/or shape
            information (<span style="color:#00C000"><b>default</b></span>: same as input).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    # (requires function as first argument)
    def __init__(
        self, name, function, output_spec=None, input_spec=None, summary_labels=None,
        l2_regularization=None
    ):
        self.output_spec = output_spec

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.function = function

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        if self.output_spec is not None:
            input_spec.update(self.output_spec)

        return input_spec

    def tf_apply(self, x):
        return self.function(x)


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
        if not isinstance(tensor, str):
            raise TensorforceError.type(name='register', argument='tensor', value=tensor)

        self.tensor = tensor

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

        Module.register_tensor(name=self.tensor, spec=self.input_spec, batched=True)

        self.output_spec = None

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def tf_apply(self, x):
        last_scope = Module.global_scope.pop()
        Module.update_tensor(name=self.tensor, tensor=x)
        Module.global_scope.append(last_scope)

        return x


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

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels, l2_regularization=0.0
        )

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
                last_scope = Module.global_scope.pop()
                x = Module.retrieve_tensor(name=self.tensors[0])
                Module.global_scope.append(last_scope)
                return x

        tensors = list()
        for tensor in self.tensors:
            if tensor == '*':
                tensors.append(x)
            else:
                last_scope = Module.global_scope.pop()
                tensors.append(Module.retrieve_tensor(name=tensor))
                Module.global_scope.append(last_scope)

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


class Reuse(Layer):
    """
    Reuse layer (specification key: `reuse`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        layer (string): Name of a previously defined layer
            (<span style="color:#C00000"><b>required</b></span>).
        is_trainable (bool): Whether reused layer variables are kept trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
    """

    def __init__(self, name, layer, is_trainable=True, input_spec=None):
        self.layer = layer
        self.is_trainable = is_trainable

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=None, l2_regularization=0.0
        )

    def default_input_spec(self):
        # from tensorforce.core.networks import Network

        # if not isinstance(self.parent, Network):
        #     raise TensorforceError.unexpected()

        # if self.layer not in self.parent.modules:
        #     raise TensorforceError.unexpected()

        # self.layer = self.parent.modules[self.layer]

        if self.layer not in Layer.layers:
            raise TensorforceError.unexpected()

        self.layer = Layer.layers[self.layer]

        return dict(self.layer.input_spec)

    def get_output_spec(self, input_spec):
        return self.layer.get_output_spec(input_spec=input_spec)

    def tf_apply(self, x):
        return self.layer.apply(x=x)

    def get_variables(self, only_trainable=False, only_saved=False):
        variables = super().get_variables(only_trainable=only_trainable, only_saved=only_saved)

        if only_trainable and self.is_trainable:
            variables.extend(self.layer.get_variables(
                only_trainable=only_trainable, only_saved=only_saved
            ))
        elif only_saved:
            pass
        else:
            variables.extend(self.layer.get_variables(
                only_trainable=only_trainable, only_saved=only_saved
            ))

        return variables

    def get_available_summaries(self):
        summaries = super().get_available_summaries()
        summaries.update(self.layer.get_available_summaries())
        return sorted(summaries)
