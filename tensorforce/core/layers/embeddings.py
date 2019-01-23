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
    Embedding layer.
    """

    def __init__(
        self, name, size, num_embeddings=None, partition_strategy='mod', max_norm=None,
        is_trainable=True, bias=False, activation='tanh', dropout=None, input_spec=None,
        l2_regularization=None, summary_labels=None
    ):
        """
        Embedding constructor.

        Args:
            num_embeddings (int > 0): Number of embeddings.
            partition_strategy ('mod' | 'div'): Partitioning strategy (see TensorFlow docs).
            max_norm (float): If not None, embeddings are clipped if their L2-norm is larger.
            is_trainable (bool):
        """
        self.num_embeddings = num_embeddings
        self.partition_strategy = partition_strategy
        self.max_norm = max_norm
        self.is_trainable = is_trainable

        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

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

        self.weights = self.add_variable(
            name='embeddings', dtype='float', shape=(self.num_embeddings, self.size),
            is_trainable=self.is_trainable, initializer='random'
        )

    def tf_apply(self, x):
        if util.tf_dtype('int') not in (tf.int32, tf.int64):
            x = tf.cast(x=x, dtype=tf.int32)
        elif util.dtype(x=x) == 'bool':
            x = tf.cast(x=x, dtype=util.tf_dtype('int'))

        x = tf.nn.embedding_lookup(
            params=self.weights, ids=x, partition_strategy=self.partition_strategy,
            max_norm=self.max_norm
        )

        return super().tf_apply(x=x)
