

class Nonlinearity(Layer):
    """
    Non-linearity layer applying a non-linear transformation.
    """

    def __init__(
        self, name, input_spec, activation='relu', alpha=None, beta=1.0, max=None, min=None,
        summary_labels=None
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
        super().__init__(
            name=name, input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

        if self.input_spec['type'] != 'float':
            raise TensorforceError(
                "Invalid input type for nonlinearity layer: {}.".format(self.input_spec['type'])
            )

        self.activation = activation
        self.alpha = None
        self.max = None
        self.min = None
        self.beta_learn = False

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

        if self.activation == 'elu':
            x = tf.nn.elu(features=(self.beta * x))

        elif self.activation == 'none':
            x = tf.identity(input=(self.beta * x))

        elif self.activation == 'relu':
            x = tf.nn.relu(features=(self.beta * x))

            non_zero = tf.cast(x=tf.count_nonzero(input_tensor=x), dtype=tf.float32)
            size = tf.cast(x=tf.reduce_prod(input_tensor=tf.shape(input=x)), dtype=tf.float32)
            self.add_summary(label='relu', name='relu', tensor=(non_zero / size))

        elif self.activation == 'selu':
            # https://arxiv.org/pdf/1706.02515.pdf
            x = tf.nn.selu(features=(self.beta * x))

        elif self.activation == 'sigmoid':
            x = tf.sigmoid(x=(self.beta * x))

        elif self.activation == 'swish':
            # https://arxiv.org/abs/1710.05941
            x = tf.sigmoid(x=(self.beta * x)) * x

        elif self.activation == 'lrelu' or self.name == 'leaky_relu':
            if self.alpha is None:
                # Default alpha value for leaky_relu
                self.alpha = 0.2
            x = tf.nn.leaky_relu(features=(self.beta * x), alpha=self.alpha)

        elif self.activation == 'crelu':
            x = tf.nn.crelu(features=(self.beta * x))

        elif self.activation == 'softmax':
            x = tf.nn.softmax(logits=(self.beta * x))

        elif self.activation == 'softplus':
            x = tf.nn.softplus(features=(self.beta * x))

        elif self.activation == 'softsign':
            x = tf.nn.softsign(features=(self.beta * x))

        elif self.activation == 'tanh':
            x = tf.nn.tanh(x=(self.beta * x))

        else:
            raise TensorforceError('Invalid activation: {}'.format(self.activation))

        self.add_summary(label='nonlinearity-beta', name='beta', tensor=self.beta)

        return x




class TFLayer(Layer):
    """
    Wrapper class for TensorFlow layers.
    """

    tensorflow_layers = dict(
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

    def __init__(self, name, layer, summary_labels=None, **kwargs):
        """
        Creates a new layer instance of a TensorFlow layer.

        Args:
            name: The name of the layer, one of 'dense'.
            **kwargs: Additional arguments passed on to the TensorFlow layer constructor.
        """
        super().__init__(name=name, summary_labels=summary_labels)

        self.layer_spec = layer
        self.layer = util.resolve_object(
            obj=layer, predefined_objects=TFLayer.tensorflow_layers, kwargs=kwargs
        )
        self.first_scope = None

    def get_output_spec(self):
        raise NotImplementedError

    def tf_apply(self, x, update):
        if self.first_scope is None:
            # Store scope of first call since regularization losses will be registered there.
            self.first_scope = tf.contrib.framework.get_name_scope()
        if isinstance(self.layer, (tf.layers.BatchNormalization, tf.layers.Dropout)):
            return self.layer(inputs=x, training=update)
        else:
            return self.layer(inputs=x)

    def tf_regularize(self):
        losses = tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.first_scope)
        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None


class Flatten(Layer):
    """
    Flatten layer reshaping the input.
    """

    def __init__(self, name, input_spec):
        super().__init__(name=name, input_spec=input_spec, l2_regularization=0.0)

        util.deprecated_module(module='Flatten', new='GlobalPooling')

    def get_output_spec(self):
        output_spec = super().get_output_spec()

        output_spec['shape'] = util.product(xs=output_spec['shape'])

        return output_spec

    def tf_apply(self, x, update):
        return tf.reshape(tensor=x, shape=(-1, util.prod(util.shape(x)[1:])))


class GlobalPooling(Layer):
    """
    Global pooling layer.
    """

    def __init__(self, name, input_spec, pooling='concat'):
        super().__init__(name=name, input_spec=input_spec, l2_regularization=0.0)

        assert pooling in ('concat', 'max', 'product', 'sum')
        self.pooling = pooling

    def get_output_spec(self):
        output_spec = super().get_output_spec()

        if self.pooling == 'concat':
            output_spec['shape'] = util.product(xs=output_spec['shape'])
        elif self.pooling == 'max':
            output_spec['shape'] = (output_spec['shape'][-1],)

        return output_spec

    def tf_apply(self, x, update):
        if self.pooling == 'concat':
            return tf.reshape(tensor=x, shape=(-1, util.prod(util.shape(x)[1:])))

        elif self.pooling == 'max':
            for _ in range(util.rank(x=x) - 2):
                x = tf.reduce_max(input_tensor=x, axis=1)
            return x

        elif self.pooling == 'product':
            for _ in range(util.rank(x=x) - 2):
                x = tf.reduce_prod(input_tensor=x, axis=1)
            return x

        elif self.pooling == 'sum':
            for _ in range(util.rank(x=x) - 2):
                x = tf.reduce_sum(input_tensor=x, axis=1)
            return x


class Pool2d(Layer):
    """
    2-dimensional pooling layer.
    """

    def __init__(self, name, input_spec, pooling='max', window=2, stride=2, padding='SAME'):
        """
        2-dimensional pooling layer.

        Args:
            pooling_type: Either 'max' or 'average'.
            window: Pooling window size, either an integer or pair of integers.
            stride: Pooling stride, either an integer or pair of integers.
            padding: Pooling padding, one of 'VALID' or 'SAME'.
        """
        super().__init__(name=name, input_spec=input_spec, l2_regularization=0.0)

        self.pooling = pooling
        if isinstance(window, int):
            self.window = (1, window, window, 1)
        elif len(window) == 2:
            self.window = (1, window[0], window[1], 1)
        else:
            raise TensorforceError('Invalid window {} for pool2d layer, must be of size 2'.format(window))
        if isinstance(stride, int):
            self.stride = (1, stride, stride, 1)
        elif len(window) == 2:
            self.stride = (1, stride[0], stride[1], 1)
        else:
            raise TensorforceError('Invalid stride {} for pool2d layer, must be of size 2'.format(stride))
        self.padding = padding

    def get_output_spec(self):
        raise NotImplementedError

    def tf_apply(self, x, update):
        if self.pooling == 'average':
            x = tf.nn.avg_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        elif self.pooling == 'max':
            x = tf.nn.max_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        else:
            raise TensorforceError('Invalid pooling type: {}'.format(self.pooling))

        return x


class Dueling(Layer):
    """
    Dueling layer, i.e. Duel pipelines for Exp & Adv to help with stability
    """

    def __init__(
        self,
        name,
        size,
        bias=False,
        activation='none',
        output=None,
        named_tensors=None,
        scope='dueling',
        l2_regularization=None,
        summary_labels=None
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
        super().__init__(
            name=name, l2_regularization=l2_regularization, summary_labels=summary_labels
        )

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
