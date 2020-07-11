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

from tensorforce.core import tf_function, tf_util
from tensorforce.core.layers import Layer


class Keras(Layer):
    """
    Keras layer (specification key: `keras`).

    Args:
        layer (string): Keras layer class name, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__
            (<span style="color:#C00000"><b>required</b></span>).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        input_spec (specification): <span style="color:#00C000"><b>internal use</b></span>.
        kwargs: Arguments for the Keras layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__.
    """

    def __init__(self, *, layer, l2_regularization=None, name=None, input_spec=None, **kwargs):
        super().__init__(l2_regularization=l2_regularization, name=name, input_spec=input_spec)

        self.keras_layer = getattr(tf.keras.layers, layer)(
            name=name, dtype=tf_util.get_dtype(type='float'), input_shape=input_spec.shape, **kwargs
        )

    def output_spec(self):
        output_spec = super().output_spec()

        output_spec.type = 'float'
        output_spec.shape = self.keras_layer.compute_output_shape(
            input_shape=((None,) + output_spec.shape)
        )[1:]

        return output_spec

    def initialize(self):
        super().initialize()

        self.keras_layer.build(input_shape=((None,) + self.input_spec.shape))

    @tf_function(num_args=0)
    def regularize(self):
        regularization_loss = super().regularize()

        if len(self.keras_layer.losses) > 0:
            regularization_loss += tf.math.add_n(inputs=self.keras_layer.losses)

        return regularization_loss

    @tf_function(num_args=1)
    def apply(self, *, x):
        x = self.keras_layer.call(inputs=x)

        return x
