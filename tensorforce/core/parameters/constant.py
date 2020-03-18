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
from tensorforce.core.parameters import Parameter


class Constant(Parameter):
    """
    Constant hyperparameter.

    Args:
        value (dtype-dependent): Constant hyperparameter value
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    # Argument 'value' first for default specification
    def __init__(
        self, value, summary_labels=None, name=None, dtype=None, min_value=None, max_value=None
    ):
        if isinstance(value, bool):
            if dtype != 'bool':
                raise TensorforceError.unexpected()
        elif isinstance(value, int):
            if dtype not in ('int', 'long'):
                raise TensorforceError.unexpected()
        elif isinstance(value, float):
            if dtype != 'float':
                raise TensorforceError.unexpected()
        else:
            raise TensorforceError.unexpected()

        self.constant_value = value

        super().__init__(
            summary_labels=summary_labels, name=name, dtype=dtype, min_value=min_value,
            max_value=max_value
        )

    def min_value(self):
        return self.constant_value

    def max_value(self):
        return self.constant_value

    def final_value(self):
        return self.constant_value

    def parameter_value(self, step):
        parameter = tf.constant(value=self.constant_value, dtype=util.tf_dtype(dtype=self.dtype))

        return parameter
