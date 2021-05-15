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
from tensorforce.core.networks import Network


class KerasNetwork(Network):
    """
    Wrapper class for networks specified as Keras model (specification key: `keras`).

    Args:
        model (tf.keras.Model): Keras model
            (<span style="color:#C00000"><b>required</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        inputs_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        outputs (iter[string]): <span style="color:#0000C0"><b>internal use</b></span>.
        kwargs: Arguments for the Keras model.
    """

    def __init__(
        self, *, model, device=None, l2_regularization=None, name=None, inputs_spec=None,
        outputs=None, **kwargs
    ):
        if outputs is not None:
            raise TensorforceError.invalid(
                name='policy', argument='single_output', condition='KerasNetwork'
            )

        super().__init__(
            device=device, l2_regularization=l2_regularization, name=name, inputs_spec=inputs_spec,
            outputs=outputs
        )

        if isinstance(model, tf.keras.Model):
            self.keras_model = model
        elif (isinstance(model, type) and issubclass(model, tf.keras.Model)):
            self.keras_model = model(**kwargs)
        elif callable(model):
            self.keras_model = model(**kwargs)
            assert isinstance(self.keras_model, tf.keras.Model)
        else:
            raise TensorforceError.value(name='KerasNetwork', argument='model', value=model)

        # if self.keras_model.inputs is not None:
        #     assert False

    def get_architecture(self):
        return 'KerasNetwork(model={})'.format(self.keras_model.__class__.__name__)

    def output_spec(self):
        assert self.keras_model.compute_dtype in (tf.float32, tf.float64)

        if self.inputs_spec.is_singleton():
            input_shape = (None,) + self.inputs_spec.singleton().shape
        else:
            input_shape = [(None,) + spec.shape for spec in self.inputs_spec.values()]

        output_shape = self.keras_model.compute_output_shape(input_shape=input_shape)
        assert isinstance(output_shape, tf.TensorShape) and output_shape.rank == 2
        output_shape = output_shape.as_list()
        assert output_shape[0] is None

        return TensorSpec(type='float', shape=(output_shape[1],))

    def initialize(self):
        super().initialize()

        if self.inputs_spec.is_singleton():
            input_shape = (None,) + self.inputs_spec.singleton().shape
        else:
            input_shape = [(None,) + spec.shape for spec in self.inputs_spec.values()]

        self.keras_model.build(input_shape=input_shape)

    @tf_function(num_args=0)
    def regularize(self):
        regularization_loss = super().regularize()

        if len(self.keras_model.losses) > 0:
            regularization_loss += tf.math.add_n(inputs=self.keras_model.losses)

        return regularization_loss

    @tf_function(num_args=4)
    def apply(self, *, x, horizons, internals, deterministic, independent):
        if x.is_singleton():
            inputs = x.singleton()
        else:
            inputs = list(x.values())

        x = self.keras_model(inputs=inputs, training=(not independent))

        return tf_util.cast(x=x, dtype='float'), internals
