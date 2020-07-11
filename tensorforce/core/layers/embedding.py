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

from tensorforce import TensorforceError
from tensorforce.core import TensorSpec, tf_function, tf_util
from tensorforce.core.layers import TransformationBase


class Embedding(TransformationBase):
    """
    Embedding layer (specification key: `embedding`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        num_embeddings (int > 0): If set, specifies the number of embeddings
            (<span style="color:#00C000"><b>default</b></span>: none).
        max_norm (float): If set, embeddings are clipped if their L2-norm is larger
            (<span style="color:#00C000"><b>default</b></span>: none).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: tanh).
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        vars_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
    """

    def __init__(
        self, *, size, num_embeddings=None, max_norm=None, bias=True, activation='tanh',
        dropout=0.0, vars_trainable=True, l2_regularization=None, name=None,
        input_spec=None
    ):
        super().__init__(
            size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable, l2_regularization=l2_regularization, name=name,
            input_spec=input_spec
        )

        self.num_embeddings = num_embeddings
        self.max_norm = max_norm

    def default_input_spec(self):
        return TensorSpec(type=('int', 'bool'), shape=None, num_values=0)

    def output_spec(self):
        output_spec = super().output_spec()

        output_spec.type = 'float'
        if not self.squeeze:
            if output_spec.shape is None:
                output_spec.shape = (None, self.size)
            else:
                output_spec.shape = output_spec.shape + (self.size,)

        return output_spec

    def initialize(self):
        super().initialize()

        if self.num_embeddings is None:
            if self.input_spec.type == 'bool':
                if self.num_embeddings is None:
                    self.num_embeddings = 2

            elif self.input_spec.type == 'int':
                if self.num_embeddings is None:
                    self.num_embeddings = self.input_spec.num_values

                if self.num_embeddings is None:
                    raise TensorforceError.required(
                        name='Embedding', argument='num_embeddings',
                        condition='input num_values is None'
                    )
                elif self.input_spec.num_values is not None and \
                        self.num_embeddings < self.input_spec.num_values:
                    raise TensorforceError.required(
                        name='Embedding', argument='num_embeddings',
                        expected='>= input num_values'
                    )

        initializer = 'normal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.variable(
            name='embeddings',
            spec=TensorSpec(type='float', shape=(self.num_embeddings, self.size)),
            initializer=initializer, is_trainable=self.vars_trainable, is_saved=True
        )

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = tf_util.int32(x=x)
        x = tf.nn.embedding_lookup(params=self.weights, ids=x, max_norm=self.max_norm)

        return super().apply(x=x)
