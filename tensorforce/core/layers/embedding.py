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
from tensorforce.core.layers import TransformationBase


class Embedding(TransformationBase):
    """
    Embedding layer (specification key: `embedding`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        num_embeddings (int > 0): If set, specifies the number of embeddings
            (<span style="color:#00C000"><b>default</b></span>: none).
        max_norm (float): If set, embeddings are clipped if their L2-norm is larger
            (<span style="color:#00C000"><b>default</b></span>: none).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: false).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: "tanh").
        dropout (parameter, 0.0 <= float < 1.0): Dropout rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).
        is_trainable (bool): Whether layer variables are trainable
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
        self, name, size, num_embeddings=None, max_norm=None, bias=False, activation='tanh',
        dropout=0.0, is_trainable=True, input_spec=None, summary_labels=None,
        l2_regularization=None
    ):
        """
        Embedding constructor.

        Args:
            size (int >= 0): Layer output size, 0 implies additionally removing the axis
                (**required**).
            bias (bool): Whether to add a trainable bias variable (default: false).
            activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
                'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
                (default: 'tanh').
        """
        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            is_trainable=is_trainable, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.num_embeddings = num_embeddings
        self.max_norm = max_norm

    def default_input_spec(self):
        return dict(type=('int', 'bool'), shape=None, num_values=0)

    def get_output_spec(self, input_spec):
        input_spec['type'] = 'float'
        if not self.squeeze:
            if input_spec['shape'] is None:
                input_spec['shape'] = (None, self.size)
            else:
                input_spec['shape'] = input_spec['shape'] + (self.size,)
        input_spec.pop('num_values', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        if self.num_embeddings is None:
            if self.input_spec['type'] == 'bool':
                self.num_embeddings = 2
            elif self.input_spec['type'] == 'int':
                self.num_embeddings = self.input_spec['num_values']
                if self.num_embeddings == 0:
                    raise TensorforceError.value(
                        name='input_spec', argument='num_values', value=self.num_embeddings
                    )

        initializer = 'normal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        self.weights = self.add_variable(
            name='embeddings', dtype='float', shape=(self.num_embeddings, self.size),
            is_trainable=self.is_trainable, initializer=initializer
        )

    def tf_apply(self, x):
        if util.tf_dtype('int') not in (tf.int32, tf.int64):
            x = tf.dtypes.cast(x=x, dtype=tf.int32)
        elif util.dtype(x=x) == 'bool':
            x = tf.dtypes.cast(x=x, dtype=util.tf_dtype('int'))

        x = tf.nn.embedding_lookup(params=self.weights, ids=x, max_norm=self.max_norm)

        return super().tf_apply(x=x)
