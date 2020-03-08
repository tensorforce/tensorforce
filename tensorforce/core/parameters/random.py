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
from tensorforce.core.parameters import Parameter


class Random(Parameter):
    """
    Random hyperparameter.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        dtype ("bool" | "int" | "long" | "float"): Tensor type
            (<span style="color:#0000C0"><b>internal use</b></span>).
        distribution ("normal" | "uniform"): Distribution type for random hyperparameter value
            (<span style="color:#C00000"><b>required</b></span>).
        shape (iter[int > 0]): Tensor shape
            (<span style="color:#0000C0"><b>internal use</b></span>).
        min_value (dtype-compatible value): Lower parameter value bound
            (<span style="color:#0000C0"><b>internal use</b></span>).
        max_value (dtype-compatible value): Upper parameter value bound
            (<span style="color:#0000C0"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments dependent on distribution type.<br>
            Normal distribution:
            <ul>
            <li><b>mean</b> (<i>float</i>) &ndash; Mean
            (<span style="color:#00C000"><b>default</b></span>: 0.0).</li>
            <li><b>stddev</b> (<i>float > 0.0</i>) &ndash; Standard deviation
            (<span style="color:#00C000"><b>default</b></span>: 1.0).</li>
            </ul>
            Uniform distribution:
            <ul>
            <li><b>minval</b> (<i>int / float</i>) &ndash; Lower bound
            (<span style="color:#00C000"><b>default</b></span>: 0 / 0.0).</li>
            <li><b>maxval</b> (<i>float > minval</i>) &ndash; Upper bound
            (<span style="color:#00C000"><b>default</b></span>: 1.0 for float,
            <span style="color:#C00000"><b>required</b></span> for int).</li>
            </ul>
        """

    def __init__(
        self, name, dtype, distribution, shape=(), min_value=None, max_value=None,
        summary_labels=None, **kwargs
    ):
        assert dtype in ('int', 'long', 'float')
        assert distribution in ('normal', 'uniform')

        self.distribution = distribution
        self.kwargs = kwargs

        super().__init__(
            name=name, dtype=dtype, shape=shape, min_value=min_value, max_value=max_value,
            summary_labels=summary_labels
        )

    def min_value(self):
        if self.distribution == 'uniform':
            if self.dtype == 'int' or self.dtype == 'long':
                return int(self.kwargs.get('minval', 0))
            elif self.dtype == 'float':
                return int(self.kwargs.get('minval', 0.0))
        else:
            return super().min_value()

    def max_value(self):
        if self.distribution == 'uniform':
            if self.dtype == 'int' or self.dtype == 'long':
                return float(self.kwargs['maxval'])
            elif self.dtype == 'float':
                return float(self.kwargs.get('maxval', 1.0))
        else:
            return super().max_value()

    def final_value(self):
        if self.distribution == 'normal':
            return util.py_dtype(dtype=self.dtype)(self.kwargs.get('mean', 0.0))

        elif self.distribution == 'uniform':
            if self.kwargs.get('maxval', None) is None:
                return 0.5

            else:
                return util.py_dtype(dtype=self.dtype)(
                    self.kwargs['maxval'] - self.kwargs.get('minval', 0)
                )

        else:
            assert False

    def parameter_value(self, step):
        if self.distribution == 'normal':
            parameter = tf.random.normal(
                shape=self.shape, dtype=util.tf_dtype(dtype=self.dtype),
                mean=self.kwargs.get('mean', 0.0), stddev=self.kwargs.get('stddev', 1.0)
            )

        elif self.distribution == 'uniform':
            parameter = tf.random.uniform(
                shape=self.shape, dtype=util.tf_dtype(dtype=self.dtype),
                minval=self.kwargs.get('minval', 0), maxval=self.kwargs.get('maxval', None)
            )

        return parameter
