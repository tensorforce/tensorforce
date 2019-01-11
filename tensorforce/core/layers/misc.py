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
from tensorforce.core import Module
from tensorforce.core.layers import Layer


class Activation(Layer):
    """
    Activation layer applying a non-linear function.
    """

    def __init__(
        self, name, nonlinearity, input_spec=None, summary_labels=None
    ):
        """
        Activation constructor.

        Args:
            nonlinearity ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
                'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Nonlinearity.
        """
        # Nonlinearity
        if nonlinearity not in (
            'crelu', 'elu', 'leaky-relu', 'none', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus',
            'softsign', 'swish', 'tanh'
        ):
            raise TensorforceError('Invalid nonlinearity: {}'.format(self.nonlinearity))
        self.nonlinearity = nonlinearity

        super().__init__(
            name=name, input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
        )

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
            # https://arxiv.org/pdf/1706.02515.pdf
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


class Dropout(Layer):
    """
    Dropout layer.
    """

    def __init__(self, name, rate, input_spec=None, summary_labels=None):
        """
        Dropout constructor.

        Args:
            dropout (0.0 <= float < 1.0): Dropout rate.
        """
        # Rate
        self.rate = rate

        super().__init__(
            name=name, input_spec=input_spec, l2_regularization=0.0, summary_labels=summary_labels
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
        def true_fn():
            dropout = tf.nn.dropout(x=x, keep_prob=(1.0 - self.rate))
            return self.add_summary(
                label='dropout', name='dropout', tensor=tf.math.zero_fraction(value=dropout),
                pass_tensors=dropout
            )

        update = Module.retrieve_tensor(name='update')
        return self.cond(pred=update, true_fn=true_fn, false_fn=(lambda: x))
