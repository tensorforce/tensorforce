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

from tensorforce.core import tf_function
from tensorforce.core.layers import TransformationBase


class Dense(TransformationBase):
    """
    Dense fully-connected layer (specification key: `dense`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        activation ('crelu' | 'elu' | 'leaky-relu' | 'none' | 'relu' | 'selu' | 'sigmoid' |
            'softmax' | 'softplus' | 'softsign' | 'swish' | 'tanh'): Activation nonlinearity
            (<span style="color:#00C000"><b>default</b></span>: "relu").
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
    """

    def default_input_spec(self):
        return dict(type='float', shape=(0,))

    def output_spec(self):
        output_spec = super().output_spec()

        if self.squeeze:
            output_spec['shape'] = output_spec['shape'][:-1]
        else:
            output_spec['shape'] = output_spec['shape'][:-1] + (self.size,)
        output_spec.pop('min_value', None)
        output_spec.pop('max_value', None)

        return output_spec

    def tf_initialize(self):
        super().tf_initialize()

        initializer = 'orthogonal'
        if self.activation is not None and self.activation.nonlinearity == 'relu':
            initializer += '-relu'

        in_size = self.input_spec['shape'][0]
        self.weights = self.add_variable(
            name='weights', dtype='float', shape=(in_size, self.size),
            is_trainable=self.is_trainable, initializer=initializer
        )

    @tf_function(num_args=1)
    def apply(self, x):
        x = tf.matmul(a=x, b=self.weights)

        return super().apply(x=x)
