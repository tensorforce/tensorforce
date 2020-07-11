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
from tensorforce.core import tf_function, TensorSpec
from tensorforce.core.layers import Conv1d, Conv2d, Dense, Layer


class Linear(Layer):
    """
    Linear layer (specification key: `linear`).

    Args:
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
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
        self, *, size, bias=True, initialization_scale=1.0, vars_trainable=True,
        l2_regularization=None, name=None, input_spec=None
    ):
        super().__init__(l2_regularization=l2_regularization, name=name, input_spec=input_spec)

        if len(self.input_spec.shape) <= 1:
            self.linear = self.submodule(
                name='linear', module=Dense, size=size, bias=bias, activation=None, dropout=0.0,
                initialization_scale=initialization_scale, vars_trainable=vars_trainable,
                input_spec=self.input_spec
            )
        elif len(self.input_spec.shape) == 2:
            self.linear = self.submodule(
                name='linear', module=Conv1d, size=size, window=1, bias=bias, activation=None,
                dropout=0.0, initialization_scale=initialization_scale,
                vars_trainable=vars_trainable, input_spec=self.input_spec
            )
        elif len(self.input_spec.shape) == 3:
            self.linear = self.submodule(
                name='linear', module=Conv2d, size=size, window=1, bias=bias, activation=None,
                dropout=0.0, initialization_scale=initialization_scale,
                vars_trainable=vars_trainable, input_spec=self.input_spec
            )
        else:
            raise TensorforceError.value(
                name='Linear', argument='input rank', value=len(self.input_spec.shape), hint='<= 3'
            )

    def default_input_spec(self):
        return TensorSpec(type='float', shape=None)

    def output_spec(self):
        return self.linear.output_spec()

    @tf_function(num_args=1)
    def apply(self, *, x):
        if len(self.input_spec.shape) == 0:
            x = tf.expand_dims(input=x, axis=1)

        x = self.linear.apply(x=x)

        return super().apply(x=x)
