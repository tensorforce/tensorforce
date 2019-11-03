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

from tensorforce import TensorforceError
import tensorforce.core
from tensorforce.core.layers import Layer


class Linear(Layer):
    """
    Linear layer (specification key: `linear`).

    Args:
        name (string): Layer name
            (<span style="color:#00C000"><b>default</b></span>: internally chosen).
        size (int >= 0): Layer output size, 0 implies additionally removing the axis
            (<span style="color:#C00000"><b>required</b></span>).
        bias (bool): Whether to add a trainable bias variable
            (<span style="color:#00C000"><b>default</b></span>: true).
        is_trainable (bool): Whether layer variables are trainable
            (<span style="color:#00C000"><b>default</b></span>: true).
        input_spec (specification): Input tensor specification
            (<span style="color:#00C000"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, size, bias=True, is_trainable=True, input_spec=None, summary_labels=None,
        l2_regularization=None
    ):
        self.size = size
        self.bias = bias
        self.is_trainable = is_trainable

        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def get_output_spec(self, input_spec):
        if len(input_spec['shape']) == 1:
            self.linear = self.add_module(
                name=(self.name + '-linear'), module='dense',
                modules=tensorforce.core.layer_modules, size=self.size, bias=self.bias,
                activation=None, dropout=0.0, is_trainable=self.is_trainable, input_spec=input_spec
            )

        elif len(input_spec['shape']) == 2:
            self.linear = self.add_module(
                name=(self.name + '-linear'), module='conv1d',
                modules=tensorforce.core.layer_modules, size=self.size, window=1, bias=self.bias,
                activation=None, dropout=0.0, is_trainable=self.is_trainable, input_spec=input_spec
            )

        elif len(input_spec['shape']) == 3:
            self.linear = self.add_module(
                name=(self.name + '-linear'), module='conv2d',
                modules=tensorforce.core.layer_modules, size=self.size, window=1, bias=self.bias,
                activation=None, dropout=0.0, is_trainable=self.is_trainable, input_spec=input_spec
            )

        else:
            raise TensorforceError.unexpected()

        return self.linear.get_output_spec(input_spec=input_spec)

    def tf_apply(self, x):
        return self.linear.apply(x=x)
