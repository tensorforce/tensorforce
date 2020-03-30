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

from tensorforce import util
from tensorforce.core import tf_function
from tensorforce.core.layers import Layer


class Keras(Layer):
    """
    Keras layer (specification key: `keras`).

    Args:
        layer (string): Keras layer class name, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__
            (<span style="color:#C00000"><b>required</b></span>).
        kwargs: Arguments for the Keras layer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__.
    """

    def __init__(
        self, name, layer, input_spec=None, summary_labels=None, l2_regularization=None, **kwargs
    ):
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.keras_layer = getattr(tf.keras.layers, layer)(
            name=name, dtype=util.tf_dtype(dtype='float'), input_shape=input_spec['shape'],
            **kwargs
        )

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def output_spec(self):
        output_spec = super.output_spec()

        shape = self.keras_layer.compute_output_shape(input_shape=((None,) + output_spec['shape']))

        return dict(type='float', shape=tuple(shape.as_list()[1:]))

    def tf_initialize(self):
        super().tf_initialize()

        self.keras_layer.build(input_shape=((None,) + self.input_spec['shape']))

    @tf_function(num_args=0)
    def regularize(self):
        if len(self.keras_layer.losses) > 0:
            regularization_loss = tf.math.add_n(inputs=self.keras_layer.losses)
        else:
            regularization_loss = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))

        return regularization_loss

    @tf_function(num_args=1)
    def apply(self, x):
        return self.keras_layer.call(inputs=x)
