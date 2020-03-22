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
from tensorforce.core import tf_function
from tensorforce.core.layers import Layer
from tensorforce.core.layers.dense import Dense
from tensorforce.core.layers.convolution import Conv1d, Conv2d


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
        super().__init__(
            name=name, input_spec=input_spec, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        if len(self.input_spec['shape']) == 1:
            self.linear = self.add_module(
                name='linear', module=Dense, size=size, bias=bias, activation=None, dropout=0.0,
                is_trainable=is_trainable, input_spec=self.input_spec
            )

        elif len(self.input_spec['shape']) == 2:
            self.linear = self.add_module(
                name='linear', module=Conv1d, size=size, window=1, bias=bias, activation=None,
                dropout=0.0, is_trainable=is_trainable, input_spec=self.input_spec
            )

        elif len(self.input_spec['shape']) == 3:
            self.linear = self.add_module(
                name='linear', module=Conv2d, size=size, window=1, bias=bias, activation=None,
                dropout=0.0, is_trainable=is_trainable, input_spec=self.input_spec
            )

        else:
            raise TensorforceError.unexpected()

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def output_spec(self):
        return self.linear.output_spec()

    @tf_function(num_args=1)
    def apply(self, x):
        return self.linear.apply(x=x)
