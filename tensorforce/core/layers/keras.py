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
        self.keras_layer = getattr(tf.keras.layers, layer)(
            name=name, dtype=util.tf_dtype(dtype='float'), input_shape=input_spec['shape'],
            **kwargs
        )

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

    def default_input_spec(self):
        return dict(type=None, shape=None)

    def get_output_spec(self, input_spec):
        shape = self.keras_layer.compute_output_shape(input_shape=((None,) + input_spec['shape']))

        return dict(type='float', shape=tuple(shape.as_list()[1:]))

    def tf_initialize(self):
        super().tf_initialize()

        self.keras_layer.build(input_shape=((None,) + self.input_spec['shape']))

        for variable in self.keras_layer.trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable
            self.trainable_variables[name] = variable
        for variable in self.keras_layer.non_trainable_weights:
            name = variable.name[variable.name.rindex(self.name + '/') + len(self.name) + 1: -2]
            self.variables[name] = variable

    def tf_regularize(self):
        regularization_loss = super().tf_regularize()

        if len(self.keras_layer.losses) > 0:
            regularization_loss += tf.math.add_n(inputs=self.keras_layer.losses)

        return regularization_loss

    def tf_apply(self, x, **kwargs):
        return self.keras_layer.call(inputs=x)
