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

from collections import Counter

import tensorflow as tf

from tensorforce import TensorforceError, util
import tensorforce.core
from tensorforce.core import parameter_modules, TensorSpec, tf_function, tf_util
from tensorforce.core.layers import Layer, NondeterministicLayer


class Activation(Layer):
    """
    Activation layer (specification key: `activation`).

    Args:
        nonlinearity ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Nonlinearity
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, nonlinearity, name=None, input_spec=None):
        super().__init__(name=name, input_spec=input_spec)

        # Nonlinearity
        if nonlinearity not in (
            'crelu', 'elu', 'leaky-relu', 'none', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus',
            'softsign', 'swish', 'tanh'
        ):
            raise TensorforceError.value(
                name='activation', argument='nonlinearity', value=nonlinearity
            )
        self.nonlinearity = nonlinearity

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    @tf_function(num_args=1)
    def apply(self, *, x):
        if self.nonlinearity == 'crelu':
            x = tf.nn.crelu(features=x)

        elif self.nonlinearity == 'elu':
            x = tf.nn.elu(features=x)

        elif self.nonlinearity == 'leaky-relu':
            # TODO: make alpha public argument
            x = tf.nn.leaky_relu(features=x, alpha=0.2)

        elif self.nonlinearity == 'none':
            pass

        elif self.nonlinearity == 'relu':
            x = tf.nn.relu(features=x)

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
        layers (iter[specification]): Layers configuration, see [layers](../modules/layers.html)
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, layers, name=None, input_spec=None):
        # TODO: handle internal states and combine with layered network
        if len(layers) == 0:
            raise TensorforceError.value(
                name='block', argument='layers', value=layers, hint='zero length'
            )

        self._input_spec = input_spec
        self.layers = list(layers)

        super().__init__(name=name, input_spec=input_spec)

    def default_input_spec(self):
        # if not isinstance(self.layers[0], Layer):
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
            self.layers[n] = self.submodule(
                name=layer_name, module=layer_spec, modules=tensorforce.core.layer_modules,
                input_spec=self._input_spec
            )
            self._input_spec = self.layers[n].output_spec()

        return self.layers[0].input_spec.copy()

    def output_spec(self):
        return self.layers[-1].output_spec()

    @tf_function(num_args=1)
    def apply(self, *, x):
        for layer in self.layers:
            x = layer.apply(x=x)

        return x


class Dropout(NondeterministicLayer):
    """
    Dropout layer (specification key: `dropout`).

    Args:
        rate (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, rate, name=None, input_spec=None):
        super().__init__(name=name, input_spec=input_spec)

        # Rate
        self.rate = self.submodule(
            name='rate', module=rate, modules=parameter_modules, dtype='float', min_value=0.0,
            max_value=1.0
        )

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    @tf_function(num_args=2)
    def apply(self, *, x, deterministic):
        if self.rate.is_constant(value=0.0):
            return x

        else:
            rate = self.rate.value()

            def no_dropout():
                return x

            def apply_dropout():
                return tf.nn.dropout(x=x, rate=rate)

            zero = tf_util.constant(value=0.0, dtype='float')
            skip_dropout = tf.math.logical_or(x=deterministic, y=tf.math.equal(x=rate, y=zero))
            return tf.cond(pred=skip_dropout, true_fn=no_dropout, false_fn=apply_dropout)


class Function(Layer):
    """
    Custom TensorFlow function layer (specification key: `function`).

    Args:
        function (lambda[x -> x]): TensorFlow function
            (<span style="color:#C00000"><b>required</b></span>).
        output_spec (specification): Output tensor specification containing type and/or shape
            information (<span style="color:#00C000"><b>default</b></span>: same as input).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    # (requires function as first argument)
    def __init__(
        self, function, output_spec=None, l2_regularization=None, name=None, input_spec=None
    ):
        super().__init__(l2_regularization=l2_regularization, name=name, input_spec=input_spec)

        self.function = function
        if output_spec is None:
            self._output_spec = None
        else:
            self._output_spec = TensorSpec(**output_spec)

    def output_spec(self):
        if self._output_spec is None:
            return super().output_spec()
        else:
            return self._output_spec

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = self.function(x)

        return x


class Reshape(Layer):
    """
    Reshape layer (specification key: `reshape`).

    Args:
        shape (<i>int | iter[int]</i>): New shape
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(self, *, shape, name=None, input_spec=None):
        super().__init__(name=name, input_spec=input_spec)

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)

    def output_spec(self):
        output_spec = super().output_spec()

        if output_spec.size != util.product(xs=self.shape):
            raise TensorforceError.value(name='Reshape', argument='shape', value=self.shape)
        output_spec.shape = self.shape

        return output_spec

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = tf.reshape(tensor=x, shape=((-1,) + self.shape))

        return x
