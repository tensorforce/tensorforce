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
from tensorforce.core import Module, parameter_modules, tf_function
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
        super().__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

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
        return dict(type='float', shape=None)

    @tf_function(num_args=1)
    def apply(self, x):
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
            raise TensorforceError.value(
                name='block', argument='layers', value=layers, hint='zero length'
            )

        self._input_spec = input_spec
        self.layers = layers

        super().__init__(name=name, input_spec=input_spec, summary_labels=None)

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
            self._input_spec = self.layers[n].output_spec()

        return dict(self.layers[0].input_spec)

    def output_spec(self):
        return self.layers[-1].output_spec()

    @tf_function(num_args=1)
    def apply(self, x):
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
        super().__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

        # Rate
        self.rate = self.add_module(
            name='rate', module=rate, modules=parameter_modules, dtype='float', min_value=0.0,
            max_value=1.0
        )

    def default_input_spec(self):
        return dict(type='float', shape=None)

    @tf_function(num_args=1)
    def apply(self, x):
        rate = self.rate.value()

        def no_dropout():
            return x

        def apply_dropout():
            dropout = tf.nn.dropout(x=x, rate=rate)
            return self.add_summary(
                label='dropout', name='dropout', tensor=tf.math.zero_fraction(value=dropout),
                pass_tensors=dropout
            )

        skip_dropout = tf.math.logical_not(x=self.global_tensor(name='deterministic'))
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
        self, function, name, output_spec=None, input_spec=None, summary_labels=None,
        l2_regularization=None
    ):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.function = function
        self._output_spec = output_spec

    def output_spec(self):
        output_spec = super().output_spec()

        if self._output_spec is not None:
            output_spec.update(self._output_spec)

        return output_spec

    @tf_function(num_args=1)
    def apply(self, x):
        return self.function(x)


class Reshape(Layer):
    """
    Reshape layer (specification key: `reshape`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        shape (<i>int | iter[int]</i>): New shape
            (<span style="color:#C00000"><b>required</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, shape, input_spec=None, summary_labels=None):
        super().__init__(name=name, input_spec=input_spec, summary_labels=summary_labels)

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def output_spec(self):
        output_spec = super().output_spec()

        if util.product(xs=output_spec['shape']) != util.product(xs=self.shape):
            raise TensorforceError.value(name='Reshape', argument='shape', value=self.shape)
        output_spec['shape'] = self.shape

        return output_spec

    @tf_function(num_args=1)
    def apply(self, x):
        x = tf.reshape(tensor=x, shape=((-1,) + self.shape))

        return x


class Reuse(Layer):
    """
    Reuse layer (specification key: `reuse`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        layer (string): Name of a previously defined layer
            (<span style="color:#C00000"><b>required</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
    """

    # _TF_MODULE_IGNORED_PROPERTIES = Module._TF_MODULE_IGNORED_PROPERTIES | {'reused_layer'}

    def __init__(self, name, layer, input_spec=None):
        # if layer not in Layer.registered_layers:
        #     raise TensorforceError.value(name='reuse', argument='layer', value=layer)

        self.layer = layer

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=None, l2_regularization=0.0
        )

    @property
    def reused_layer(self):
        module = self
        while isinstance(module, Layer):
            module = module.parent
        assert isinstance(module, (tensorforce.core.networks.LayerbasedNetwork))
        return module.registered_layers[self.layer]

    def default_input_spec(self):
        return dict(self.reused_layer.input_spec)

    def output_spec(self):
        return self.reused_layer.output_spec()

    @tf_function(num_args=1)
    def apply(self, x):
        return self.reused_layer.apply(x=x)

    def get_available_summaries(self):
        summaries = super().get_available_summaries()
        summaries.update(self.reused_layer.get_available_summaries())
        return sorted(summaries)
