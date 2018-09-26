# Copyright 2017 reinforce.io. All Rights Reserved.
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

"""
Collection of custom layer implementations.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from math import sqrt
import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, util
import tensorforce.core.networks


class Layer(object):
    """
    Base class for network layers.
    """

    def __init__(self, named_tensors=None, scope='layer', summary_labels=None):
        """
        Layer.
        """
        self.scope = scope
        self.summary_labels = set(summary_labels or ())

        self.named_tensors = named_tensors
        self.variables = dict()
        self.all_variables = dict()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if registered:
                pass
            elif name in self.all_variables:
                assert variable is self.all_variables[name]
                if kwargs.get('trainable', True):
                    assert variable is self.variables[name]
                    if 'variables' in self.summary_labels:
                        tf.contrib.summary.histogram(name=name, tensor=variable)
            else:
                self.all_variables[name] = variable
                if kwargs.get('trainable', True):
                    self.variables[name] = variable
                    if 'variables' in self.summary_labels:
                        tf.contrib.summary.histogram(name=name, tensor=variable)
            return variable

        self.apply = tf.make_template(
            name_=(scope + '/apply'),
            func_=self.tf_apply,
            custom_getter_=custom_getter
        )
        self.regularization_loss = tf.make_template(
            name_=(scope + '/regularization-loss'),
            func_=self.tf_regularization_loss,
            custom_getter_=custom_getter
        )

    def tf_apply(self, x, update):
        """
        Creates the TensorFlow operations for applying the layer to the given input.

        Args:
            x: Layer input tensor.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Layer output tensor.
        """
        raise NotImplementedError

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the layer regularization loss.

        Returns:
            Regularization loss tensor.
        """
        return None

    def internals_spec(self):
        """
        Returns the internal states specification.

        Returns:
            Internal states specification
        """
        return dict()

    def get_variables(self, include_nontrainable=False):
        """
        Returns the TensorFlow variables used by the layer.

        Returns:
            List of variables.
        """
        if include_nontrainable:
            return [self.all_variables[key] for key in sorted(self.all_variables)]
        else:
            return [self.variables[key] for key in sorted(self.variables)]

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a layer from a specification dict.
        """
        layer = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.networks.layers,
            kwargs=kwargs
        )
        assert isinstance(layer, Layer)
        return layer


class Input(Layer):
    """
    Input layer. Used to collect data together as a form of output to the next layer.
    Allows for multiple inputs to merge into a single import for next layer.
    """

    def __init__(
        self,
        names,
        aggregation_type='concat',
        axis=1,
        named_tensors=None,
        scope='input',
        summary_labels=()
    ):
        """
        Input layer.

        Args:
            names: A list of strings that name the inputs to merge
            axis: Axis to merge the inputs

        """
        self.names = names
        self.aggregation_type = aggregation_type
        self.axis = axis
        super(Input, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if isinstance(self.names, str):
            if self.names == '*' or self.names == 'previous':
                # like normal list network_spec
                return x
            elif self.names in self.named_tensors:
                return self.named_tensors[self.names]
            else:
                keys = sorted(self.named_tensors)
                raise TensorForceError(
                    'Input "{}" doesn\'t exist, Available inputs: {}'.format(self.names, keys)
                )

        inputs = list()
        max_shape = ()
        for name in self.names:
            if name == '*' or name == 'previous':
                # like normal list network_spec
                tensor = x
            elif name in self.named_tensors:
                tensor = self.named_tensors[name]
            else:
                keys = sorted(self.named_tensors)
                raise TensorForceError(
                    'Input "{}" doesn\'t exist, Available inputs: {}'.format(name, keys)
                )
            inputs.append(tensor)
            shape = util.shape(x=tensor)
            if len(shape) > len(max_shape):
                max_shape = shape

        for n, tensor in enumerate(inputs):
            shape = util.shape(x=tensor)
            if len(shape) < len(max_shape):
                # assert shape == max_shape[:len(shape)], (shape, max_shape)
                for i in range(len(shape), len(max_shape)):
                    # assert max_shape[i] == 1, (shape, max_shape)
                    tensor = tf.expand_dims(input=tensor, axis=i)
                inputs[n] = tensor
            # else:
            #     assert shape == max_shape, (shape, max_shape)

        if self.aggregation_type == 'concat':
            x = tf.concat(values=inputs, axis=self.axis)
        elif self.aggregation_type == 'stack':
            x = tf.stack(values=inputs, axis=self.axis)
        elif self.aggregation_type == 'sum':
            x = tf.stack(values=inputs, axis=self.axis)
            x = tf.reduce_sum(input_tensor=x, axis=self.axis)
        elif self.aggregation_type == 'product':
            x = tf.stack(values=inputs, axis=self.axis)
            x = tf.reduce_prod(input_tensor=x, axis=self.axis)
        else:
            raise NotImplementedError

        return x


class Output(Layer):
    """
    Output layer. Used to capture the tensor under and name for use with Input layers.
    Acts as a input to output passthrough.
    """

    def __init__(
        self,
        name,
        named_tensors=None,
        scope='output',
        summary_labels=()
    ):
        """
        Output layer.

        Args:
            output: A string that names the tensor, will be added to available inputs

        """
        self.name = name
        super(Output, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        self.named_tensors[self.name] = x
        return x


class TFLayer(Layer):
    """
    Wrapper class for TensorFlow layers.
    """

    tf_layers = dict(
        average_pooling1d=tf.layers.AveragePooling1D,
        average_pooling2d=tf.layers.AveragePooling2D,
        average_pooling3d=tf.layers.AveragePooling3D,
        batch_normalization=tf.layers.BatchNormalization,
        conv1d=tf.layers.Conv1D,
        conv2d=tf.layers.Conv2D,
        conv2d_transpose=tf.layers.Conv2DTranspose,
        conv3d=tf.layers.Conv3D,
        conv3d_transpose=tf.layers.Conv3DTranspose,
        dense=tf.layers.Dense,
        dropout=tf.layers.Dropout,
        flatten=tf.layers.Flatten,
        max_pooling1d=tf.layers.MaxPooling1D,
        max_pooling2d=tf.layers.MaxPooling2D,
        max_pooling3d=tf.layers.MaxPooling3D,
        separable_conv2d=tf.layers.SeparableConv2D
    )

    def __init__(self, layer, named_tensors=None, scope='tf-layer', summary_labels=(), **kwargs):
        """
        Creates a new layer instance of a TensorFlow layer.

        Args:
            name: The name of the layer, one of 'dense'.
            **kwargs: Additional arguments passed on to the TensorFlow layer constructor.
        """
        self.layer_spec = layer
        self.layer = util.get_object(obj=layer, predefined_objects=TFLayer.tf_layers, kwargs=kwargs)
        self.first_scope = None

        super(TFLayer, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if self.first_scope is None:
            # Store scope of first call since regularization losses will be registered there.
            self.first_scope = tf.contrib.framework.get_name_scope()
        if isinstance(self.layer, (tf.layers.BatchNormalization, tf.layers.Dropout)):
            return self.layer(inputs=x, training=update)
        else:
            return self.layer(inputs=x)

    def tf_regularization_loss(self):
        regularization_losses = tf.get_collection(
            key=tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self.first_scope
        )
        if len(regularization_losses) > 0:
            return tf.add_n(inputs=regularization_losses)
        else:
            return None


class Nonlinearity(Layer):
    """
    Non-linearity layer applying a non-linear transformation.
    """

    def __init__(self,
        name='relu',
        alpha=None,
        beta=1.0,
        max=None,
        min=None,
        named_tensors=None,
        scope='nonlinearity',
        summary_labels=()
    ):
        """
        Non-linearity activation layer.

        Args:
            name: Non-linearity name, one of 'elu', 'relu', 'selu', 'sigmoid', 'swish',
                'leaky_relu' (or 'lrelu'), 'crelu', 'softmax', 'softplus', 'softsign', 'tanh' or 'none'.
            alpha: (float|int) Alpha value for leaky Relu
            beta: (float|int|'learn') Beta value or 'learn' to train value (default 1.0)
            max: (float|int) maximum (beta * input) value passed to non-linearity function
            min: (float|int) minimum (beta * input) value passed to non-linearity function
            summary_labels: Requested summary labels for tensorboard export, add 'beta' to watch beta learning
        """
        self.name = name
        self.alpha = None
        self.max = None
        self.min = None
        self.beta_learn = False
        super(Nonlinearity, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

        if max is not None:
            self.max = float(max)

        if min is not None:
            self.min = float(min)

        if alpha is not None:
            self.alpha = float(alpha)

        if beta == 'learn':
            self.beta_learn = True
            self.beta = None
        else:
            self.beta = tf.constant(float(beta), dtype=util.tf_dtype('float'))

    def tf_apply(self, x, update):
        if self.beta_learn:
            self.beta = tf.get_variable(
                name='beta',
                shape=(),
                dtype=tf.float32,
                initializer=tf.ones_initializer()
            )

        if self.max is not None:
            x = tf.minimum(x=(self.beta * x), y=self.max)

        if self.min is not None:
            x = tf.maximum(x=(self.beta * x), y=self.min)

        if self.name == 'elu':
            x = tf.nn.elu(features=(self.beta * x))

        elif self.name == 'none':
            x = tf.identity(input=(self.beta * x))

        elif self.name == 'relu':
            x = tf.nn.relu(features=(self.beta * x))
            if 'relu' in self.summary_labels:
                non_zero = tf.cast(x=tf.count_nonzero(input_tensor=x), dtype=tf.float32)
                size = tf.cast(x=tf.reduce_prod(input_tensor=tf.shape(input=x)), dtype=tf.float32)
                tf.contrib.summary.scalar(name='relu', tensor=(non_zero / size))

        elif self.name == 'selu':
            # https://arxiv.org/pdf/1706.02515.pdf
            x = tf.nn.selu(features=(self.beta * x))

        elif self.name == 'sigmoid':
            x = tf.sigmoid(x=(self.beta * x))

        elif self.name == 'swish':
            # https://arxiv.org/abs/1710.05941
            x = tf.sigmoid(x=(self.beta * x)) * x

        elif self.name == 'lrelu' or self.name == 'leaky_relu':
            if self.alpha is None:
                # Default alpha value for leaky_relu
                self.alpha = 0.2
            x = tf.nn.leaky_relu(features=(self.beta * x), alpha=self.alpha)

        elif self.name == 'crelu':
            x = tf.nn.crelu(features=(self.beta * x))

        elif self.name == 'softmax':
            x = tf.nn.softmax(logits=(self.beta * x))

        elif self.name == 'softplus':
            x = tf.nn.softplus(features=(self.beta * x))

        elif self.name == 'softsign':
            x = tf.nn.softsign(features=(self.beta * x))

        elif self.name == 'tanh':
            x = tf.nn.tanh(x=(self.beta * x))

        else:
            raise TensorForceError('Invalid non-linearity: {}'.format(self.name))

        if 'beta' in self.summary_labels:
            tf.contrib.summary.scalar(name='beta', tensor=self.beta)

        return x


class Dropout(Layer):
    """
    Dropout layer. If using dropout, add this layer after inputs and after dense layers. For
    LSTM, dropout is handled independently as an argument. Not available for Conv2d yet.
    """

    def __init__(self, rate=0.0, named_tensors=None, scope='dropout', summary_labels=()):
        self.rate = rate
        super(Dropout, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        dropout = tf.nn.dropout(x=x, keep_prob=(1.0 - self.rate))
        return tf.where(condition=update, x=dropout, y=x)


class Flatten(Layer):
    """
    Flatten layer reshaping the input.
    """

    def __init__(self, named_tensors=None, scope='flatten', summary_labels=()):
        super(Flatten, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        return tf.reshape(tensor=x, shape=(-1, util.prod(util.shape(x)[1:])))


class Pool2d(Layer):
    """
    2-dimensional pooling layer.
    """

    def __init__(
        self,
        pooling_type='max',
        window=2,
        stride=2,
        padding='SAME',
        named_tensors=None,
        scope='pool2d',
        summary_labels=()
    ):
        """
        2-dimensional pooling layer.

        Args:
            pooling_type: Either 'max' or 'average'.
            window: Pooling window size, either an integer or pair of integers.
            stride: Pooling stride, either an integer or pair of integers.
            padding: Pooling padding, one of 'VALID' or 'SAME'.
        """
        self.pooling_type = pooling_type
        if isinstance(window, int):
            self.window = (1, window, window, 1)
        elif len(window) == 2:
            self.window = (1, window[0], window[1], 1)
        else:
            raise TensorForceError('Invalid window {} for pool2d layer, must be of size 2'.format(window))
        if isinstance(stride, int):
            self.stride = (1, stride, stride, 1)
        elif len(window) == 2:
            self.stride = (1, stride[0], stride[1], 1)
        else:
            raise TensorForceError('Invalid stride {} for pool2d layer, must be of size 2'.format(stride))
        self.padding = padding
        super(Pool2d, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if self.pooling_type == 'average':
            x = tf.nn.avg_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        elif self.pooling_type == 'max':
            x = tf.nn.max_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        else:
            raise TensorForceError('Invalid pooling type: {}'.format(self.name))

        return x


class Embedding(Layer):
    """
    Embedding layer.
    """

    def __init__(
        self,
        indices,
        size,
        l2_regularization=0.0,
        l1_regularization=0.0,
        named_tensors=None,
        scope='embedding',
        summary_labels=()
    ):
        """
        Embedding layer.

        Args:
            indices: Number of embedding indices.
            size: Embedding size.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        self.indices = indices
        self.size = size
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        super(Embedding, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        stddev = min(0.1, sqrt(1.0 / self.size))
        weights_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.weights = tf.get_variable(
            name='embeddings',
            shape=(self.indices, self.size),
            dtype=tf.float32,
            initializer=weights_init
        )
        return tf.nn.embedding_lookup(params=self.weights, ids=x)

    def tf_regularization_loss(self):
        regularization_loss = super(Embedding, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.weights))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.weights)))

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None


class Linear(Layer):
    """
    Linear fully-connected layer.
    """

    def __init__(
        self,
        size,
        weights=None,
        bias=True,
        l2_regularization=0.0,
        l1_regularization=0.0,
        trainable=True,
        named_tensors=None,
        scope='linear',
        summary_labels=()
    ):
        """
        Linear layer.

        Args:
            size: Layer size.
            weights: Weight initialization, random if None.
            bias: Bias initialization, random if True, no bias added if False.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        self.size = size
        self.weights_init = weights
        self.bias_init = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.trainable = trainable
        super(Linear, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update=False):
        if util.rank(x) != 2:
            raise TensorForceError(
                'Invalid input rank for linear layer: {}, must be 2.'.format(util.rank(x))
            )

        if self.size is None:  # If size is None than Output Matches Input, required for Skip Connections
            self.size = x.shape[1].value

        weights_shape = (x.shape[1].value, self.size)

        if self.weights_init is None:
            stddev = min(0.1, sqrt(2.0 / (x.shape[1].value + self.size)))
            self.weights_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)

        elif isinstance(self.weights_init, dict):
            if 'name' in self.weights_init:
                if self.weights_init['name'] == 'msra':
                    slope = 0.25
                    if 'slope' in self.weights_init:
                        slope = self.weights_init['slope']
                    magnitude = 2.0 / (1.0 + slope ** 2)
                    stddev = sqrt(magnitude * 2.0 / (x.shape[1].value + self.size))
                    self.weights_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
            else:
                raise TensorForceError(
                    'Linear weights init with dict does not has name attribute, weight_init={}'.format(self.weights_init)
                )

        elif isinstance(self.weights_init, float):
            if self.weights_init == 0.0:
                self.weights_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, list):
            self.weights_init = np.asarray(self.weights_init, dtype=np.float32)
            if self.weights_init.shape != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights_init.shape, weights_shape)
                )
            self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, np.ndarray):
            if self.weights_init.shape != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights_init.shape, weights_shape)
                )
            self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, tf.Tensor):
            if util.shape(self.weights_init) != weights_shape:
                raise TensorForceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights_init.shape, weights_shape)
                )

        bias_shape = (self.size,)

        if isinstance(self.bias_init, bool):
            if self.bias_init:
                self.bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.bias_init = None

        elif isinstance(self.bias_init, float):
            if self.bias_init == 0.0:
                self.bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias_init, list):
            self.bias_init = np.asarray(self.bias_init, dtype=np.float32)
            if self.bias_init.shape != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias_init.shape, bias_shape)
                )
            self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias_init, np.ndarray):
            if self.bias_init.shape != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias_init.shape, bias_shape)
                )
            self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias_init, tf.Tensor):
            if util.shape(self.bias_init) != bias_shape:
                raise TensorForceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias_init.shape, bias_shape)
                )

        if isinstance(self.weights_init, tf.Tensor):
            self.weights = self.weights_init
        else:
            self.weights = tf.get_variable(
                name='W',
                shape=weights_shape,
                dtype=tf.float32,
                initializer=self.weights_init,
                trainable=self.trainable
            )

        x = tf.matmul(a=x, b=self.weights)

        if self.bias_init is None:
            self.bias = None

        else:
            if isinstance(self.bias_init, tf.Tensor):
                self.bias = self.bias_init
            else:
                self.bias = tf.get_variable(
                    name='b',
                    shape=bias_shape,
                    dtype=tf.float32,
                    initializer=self.bias_init,
                    trainable=self.trainable)

            x = tf.nn.bias_add(value=x, bias=self.bias)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Linear, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.weights))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.weights)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None


class Dense(Layer):
    """
    Dense layer, i.e. linear fully connected layer with subsequent non-linearity.
    """

    def __init__(
        self,
        size=None,
        weights=None,
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        skip=False,
        trainable=True,
        named_tensors=None,
        scope='dense',
        summary_labels=(),
    ):
        """
        Dense layer.

        Args:
            size: Layer size, if None than input size matches the output size of the layer
            weights: Weight initialization, random if None.
            bias: If true, bias is added.
            activation: Type of nonlinearity, or dict with name & arguments
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
            skip: Add skip connection like ResNet (https://arxiv.org/pdf/1512.03385.pdf),
                  doubles layers and ShortCut from Input to output
        """
        self.skip = skip
        if self.skip and size is not None:
            raise TensorForceError(
                'Dense Layer SKIP connection needs Size=None, uses input shape '
                'sizes to create skip connection network, please delete "size" parameter'
            )

        self.linear = Linear(
            size=size,
            weights=weights,
            bias=bias,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels,
            trainable=trainable
        )
        if self.skip:
            self.linear_skip = Linear(
                size=size,
                bias=bias,
                l2_regularization=l2_regularization,
                l1_regularization=l1_regularization,
                summary_labels=summary_labels,
                trainable=trainable
            )
        # TODO: Consider creating two nonlinearity variables when skip is used and learning beta
        #       Right now, only a single beta can be learned
        self.nonlinearity = Nonlinearity(summary_labels=summary_labels, **util.prepare_kwargs(activation))
        super(Dense, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        xl1 = self.linear.apply(x=x, update=update)
        xl1 = self.nonlinearity.apply(x=xl1, update=update)
        if self.skip:
            xl2 = self.linear_skip.apply(x=xl1, update=update)
            xl2 = self.nonlinearity.apply(x=(xl2 + x), update=update)  #add input back in as skip connection per paper
        else:
            xl2 = xl1

        if 'activations' in self.summary_labels:
            tf.contrib.summary.histogram(name='activations', tensor=xl2)

        return xl2

    def tf_regularization_loss(self):
        regularization_loss = super(Dense, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.linear.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if self.skip:
            regularization_loss = self.linear_skip.regularization_loss()
            if regularization_loss is not None:
                losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        layer_variables = super(Dense, self).get_variables(include_nontrainable=include_nontrainable)
        linear_variables = self.linear.get_variables(include_nontrainable=include_nontrainable)
        if self.skip:
            linear_variables = linear_variables \
                               + self.linear_skip.get_variables(include_nontrainable=include_nontrainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_nontrainable=include_nontrainable)

        return layer_variables + linear_variables + nonlinearity_variables


class Dueling(Layer):
    """
    Dueling layer, i.e. Duel pipelines for Exp & Adv to help with stability
    """

    def __init__(
        self,
        size,
        bias=False,
        activation='none',
        l2_regularization=0.0,
        l1_regularization=0.0,
        output=None,
        named_tensors=None,
        scope='dueling',
        summary_labels=()
    ):
        """
        Dueling layer.

        [Dueling Networks] (https://arxiv.org/pdf/1511.06581.pdf)
        Implement Y = Expectation[x] + (Advantage[x] - Mean(Advantage[x]))

        Args:
            size: Layer size.
            bias: If true, bias is added.
            activation: Type of nonlinearity, or dict with name & arguments
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
            output: None or tuple of output names for ('expectation','advantage','mean_advantage')
        """
        # Expectation is broadcast back over advantage values so output is of size 1
        self.expectation_layer = Linear(
            size=1, bias=bias,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels,
        )
        self.advantage_layer = Linear(
            size=size,
            bias=bias,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels,
        )
        self.output = output
        self.nonlinearity = Nonlinearity(summary_labels=summary_labels, **util.prepare_kwargs(activation))
        super(Dueling, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        expectation = self.expectation_layer.apply(x=x, update=update)
        advantage = self.advantage_layer.apply(x=x, update=update)
        mean_advantage = tf.reduce_mean(input_tensor=advantage, axis=1, keep_dims=True)

        # Record outputs in named tensor dictionary if passed
        if type(self.output) is tuple and len(self.output) == 3:
            if self.named_tensors is not None:
                self.named_tensors[self.output[0]] = expectation
                self.named_tensors[self.output[1]] = advantage - mean_advantage
                self.named_tensors[self.output[2]] = mean_advantage
            if 'activations' in self.summary_labels:
                tf.contrib.summary.histogram(name=self.output[0], tensor=expectation)
                tf.contrib.summary.histogram(name=self.output[1], tensor=advantage - mean_advantage)
                tf.contrib.summary.histogram(name=self.output[2], tensor=mean_advantage)

        x = expectation + advantage - mean_advantage

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            tf.contrib.summary.histogram(name='activations', tensor=x)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Dueling, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.expectation_layer.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.advantage_layer.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        layer_variables = super(Dueling, self).get_variables(include_nontrainable=include_nontrainable)
        expectation_layer_variables = self.expectation_layer.get_variables(include_nontrainable=include_nontrainable)
        advantage_layer_variables = self.advantage_layer.get_variables(include_nontrainable=include_nontrainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_nontrainable=include_nontrainable)

        return layer_variables + expectation_layer_variables + advantage_layer_variables + nonlinearity_variables


class Conv1d(Layer):
    """
    1-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        named_tensors=None,
        scope='conv1d',
        summary_labels=()
    ):
        """
        1D convolutional layer.

        Args:
            size: Number of filters
            window: Convolution window size
            stride: Convolution stride
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity, or dict with name & arguments
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        self.window = window
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = Nonlinearity(summary_labels=summary_labels, **util.prepare_kwargs(activation))
        super(Conv1d, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if util.rank(x) != 3:
            raise TensorForceError('Invalid input rank for conv1d layer: {}, must be 3'.format(util.rank(x)))

        filters_shape = (self.window, x.shape[2].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        x = tf.nn.conv1d(value=x, filters=self.filters, stride=self.stride, padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            tf.contrib.summary.histogram(name='activations', tensor=x)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Conv1d, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        layer_variables = super(Conv1d, self).get_variables(include_nontrainable=include_nontrainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_nontrainable=include_nontrainable)

        return layer_variables + nonlinearity_variables


class Conv2d(Layer):
    """
    2-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        named_tensors=None,
        scope='conv2d',
        summary_labels=()
    ):
        """
        2D convolutional layer.

        Args:
            size: Number of filters
            window: Convolution window size, either an integer or pair of integers.
            stride: Convolution stride, either an integer or pair of integers.
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity, or dict with name & arguments
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        if isinstance(window, int):
            self.window = (window, window)
        elif len(window) == 2:
            self.window = tuple(window)
        else:
            raise TensorForceError('Invalid window {} for conv2d layer, must be of size 2'.format(window))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = Nonlinearity(summary_labels=summary_labels, **util.prepare_kwargs(activation))
        super(Conv2d, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if util.rank(x) != 4:
            raise TensorForceError('Invalid input rank for conv2d layer: {}, must be 4'.format(util.rank(x)))

        filters_shape = self.window + (x.shape[3].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        stride_h, stride_w = self.stride if type(self.stride) is tuple else (self.stride, self.stride)
        x = tf.nn.conv2d(input=x, filter=self.filters, strides=(1, stride_h, stride_w, 1), padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            tf.contrib.summary.histogram(name='activations', tensor=x)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Conv2d, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        layer_variables = super(Conv2d, self).get_variables(include_nontrainable=include_nontrainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_nontrainable=include_nontrainable)

        return layer_variables + nonlinearity_variables


class InternalLstm(Layer):
    """
    Long short-term memory layer for internal state management.
    """

    def __init__(self, size, dropout=None, lstmcell_args={}, named_tensors=None, scope='internal_lstm', summary_labels=()):
        """
        LSTM layer.

        Args:
            size: LSTM size.
            dropout: Dropout rate.
        """
        self.size = size
        self.dropout = dropout
        self.lstmcell_args = lstmcell_args
        super(InternalLstm, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update, state):
        if util.rank(x) != 2:
            raise TensorForceError(
                'Invalid input rank for internal lstm layer: {}, must be 2.'.format(util.rank(x))
            )

        state = tf.contrib.rnn.LSTMStateTuple(c=state[:, 0, :], h=state[:, 1, :])

        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.size, **self.lstmcell_args)

        if self.dropout is not None:
            keep_prob = tf.cond(pred=update, true_fn=(lambda: 1.0 - self.dropout), false_fn=(lambda: 1.0))
            self.lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell, output_keep_prob=keep_prob)

        x, state = self.lstm_cell(inputs=x, state=state)

        state = tf.stack(values=(state.c, state.h), axis=1)

        if 'activations' in self.summary_labels:
            tf.contrib.summary.histogram(name='activations', tensor=x)

        return x, dict(state=state)

    def internals_spec(self):
        return dict(state=dict(
            type='float',
            shape=(2, self.size),
            initialization='zeros'
        ))


class Lstm(Layer):

    def __init__(self, size, dropout=None, named_tensors=None, scope='lstm', summary_labels=(), return_final_state=True):
        """
        LSTM layer.

        Args:
            size: LSTM size.
            dropout: Dropout rate.
        """
        self.size = size
        self.dropout = dropout
        self.return_final_state = return_final_state
        super(Lstm, self).__init__(named_tensors=named_tensors, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update, sequence_length=None):
        if util.rank(x) != 3:
            raise TensorForceError('Invalid input rank for lstm layer: {}, must be 3.'.format(util.rank(x)))

        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.size)
        if 'activations' in self.summary_labels:
            tf.contrib.summary.histogram(name='activations', tensor=x)

        x, state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=x,
            sequence_length=sequence_length,
            dtype=tf.float32
        )

        # This distinction is so we can stack multiple LSTM layers
        if self.return_final_state:
            return tf.concat(values=(state.c, state.h), axis=1)
        else:
            return x
