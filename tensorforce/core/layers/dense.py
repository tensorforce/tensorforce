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

from tensorforce.core import TensorSpec, tf_function
from tensorforce.core.layers import TransformationBase


class Dense(TransformationBase):
    """
    Dense fully-connected layer (specification key: `dense`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: tanh).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        initialization_scale (float > 0.0): Initialization scale
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, size, bias=True, activation='tanh', dropout=0.0, initialization_scale=1.0,
        vars_trainable=True, l2_regularization=None, name=None, input_spec=None
    ):
        super().__init__(
            size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable, l2_regularization=l2_regularization, name=name,
            input_spec=input_spec
        )

        self.initialization_scale = initialization_scale

    def default_input_spec(self):
        return TensorSpec(type='float', shape=(0,))

    def output_spec(self):
        output_spec = super().output_spec()

        if self.squeeze:
            output_spec.shape = output_spec.shape[:-1]
        else:
            output_spec.shape = output_spec.shape[:-1] + (self.size,)

        output_spec.min_value = None
        output_spec.max_value = None

        return output_spec

    def initialize(self):
        super().initialize()

        in_size = self.input_spec.shape[0]

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.variable(
            name='weights', spec=TensorSpec(type='float', shape=(in_size, self.size)),
            initializer=initializer, initialization_scale=self.initialization_scale,
            is_trainable=self.vars_trainable, is_saved=True
        )

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = tf.matmul(a=x, b=self.weights)

        return super().apply(x=x)
