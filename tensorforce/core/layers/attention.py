# Copyright 2021 Tensorforce Team. All Rights Reserved.
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

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import TensorSpec, tf_function, tf_util
from tensorforce.core.layers import Linear, TransformationBase


class SelfAttention(TransformationBase):
    """
    Self-attention layer (specification key: `self_attention`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        attention_size (int > 0): Query/key size
            (<span style="color:#00C000"><b>default</b></span>: same as output size).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: false).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: none).
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
        self, *, size, attention_size=None, bias=False, activation=None, dropout=0.0,
        initialization_scale=1.0, vars_trainable=True, l2_regularization=None, name=None,
        input_spec=None
    ):
        super().__init__(
            size=size, bias=bias, activation=activation, dropout=dropout,
            vars_trainable=vars_trainable, l2_regularization=l2_regularization, name=name,
            input_spec=input_spec
        )

        self.attention_size = size if attention_size is None else attention_size

        if input_spec.rank <= 1:
            raise TensorforceError.value(
                name='SelfAttention', argument='input_spec[shape]', value=str(input_spec.shape),
                hint='is not rank >= 2'
            )

        self.query = self.submodule(
            name='query', module=Linear, size=self.attention_size, bias=bias,
            vars_trainable=vars_trainable, input_spec=input_spec
        )
        self.key = self.submodule(
            name='key', module=Linear, size=self.attention_size, bias=bias,
            vars_trainable=vars_trainable, input_spec=input_spec
        )
        self.value = self.submodule(
            name='value', module=Linear, size=size, bias=bias,
            initialization_scale=initialization_scale, vars_trainable=vars_trainable,
            input_spec=input_spec
        )

        self.architecture_kwargs['size'] = str(size)
        self.architecture_kwargs['attention_size'] = str(self.attention_size)
        self.architecture_kwargs['bias'] = str(bias)
        if activation is not None:
            self.architecture_kwargs['activation'] = str(activation)
        if dropout != 0.0:
            self.architecture_kwargs['dropout'] = str(dropout)
        if initialization_scale != 1.0:
            self.architecture_kwargs['initialization_scale'] = str(initialization_scale)
        if not vars_trainable:
            self.architecture_kwargs['trainable'] = str(vars_trainable)
        if l2_regularization is not None:
            self.architecture_kwargs['l2_regularization'] = str(l2_regularization)

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    def output_spec(self):
        return self.value.output_spec()

    @tf_function(num_args=1)
    def apply(self, *, x):
        queries = self.query.apply(x=x)
        keys = self.key.apply(x=x)
        values = self.value.apply(x=x)

        if self.input_spec.rank > 2:
            batch_size = tf_util.cast(x=tf.shape(input=x)[:1], dtype='int')

            flattened_shape = tf_util.constant(
                value=(util.product(xs=self.input_spec.shape[:-1]), self.attention_size),
                dtype='int'
            )
            flattened_shape = tf.concat(values=(batch_size, flattened_shape), axis=0)
            queries = tf.reshape(tensor=queries, shape=flattened_shape)
            keys = tf.reshape(tensor=keys, shape=flattened_shape)

            flattened_shape = tf_util.constant(
                value=(util.product(xs=self.input_spec.shape[:-1]), self.size), dtype='int'
            )
            flattened_shape = tf.concat(values=(batch_size, flattened_shape), axis=0)
            values = tf.reshape(tensor=values, shape=flattened_shape)

        attention = tf.linalg.matmul(a=queries, b=keys, transpose_b=True)
        attention = attention / tf_util.constant(value=np.sqrt(self.attention_size), dtype='float')
        attention = tf.nn.softmax(logits=attention, axis=-1)
        x = tf.linalg.matmul(a=attention, b=values)

        if self.input_spec.rank > 2:
            shape = tf_util.constant(value=self.output_spec().shape, dtype='int')
            shape = tf.concat(values=(batch_size, shape), axis=0)
            x = tf.reshape(tensor=x, shape=shape)

        return super().apply(x=x)
