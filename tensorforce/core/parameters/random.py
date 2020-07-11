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

from tensorforce.core.parameters import Parameter


class Random(Parameter):
    """
    Random hyperparameter (specification key: `random`).

    Args:
        distribution ("normal" | "uniform"): Distribution type for random hyperparameter value
            (<span style="color:#C00000"><b>required</b></span>).
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
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        shape (iter[int > 0]): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, distribution, name=None, dtype=None, shape=(), min_value=None, max_value=None,
        **kwargs
    ):
        assert dtype in ('int', 'float')
        assert distribution in ('normal', 'uniform')

        self.distribution = distribution
        self.kwargs = kwargs

        super().__init__(
            name=name, dtype=dtype, shape=shape, min_value=min_value, max_value=max_value
        )

    def min_value(self):
        if self.distribution == 'uniform':
            return self.spec.py_type()(self.kwargs.get('minval', 0))

        else:
            return super().min_value()

    def max_value(self):
        if self.distribution == 'uniform':
            return self.spec.py_type()(self.kwargs.get('maxval', 1.0))

        else:
            return super().max_value()

    def final_value(self):
        if self.distribution == 'normal':
            return self.spec.py_type()(self.kwargs.get('mean', 0.0))

        elif self.distribution == 'uniform':
            return self.spec.py_type()(
                (self.kwargs.get('maxval', 1.0) + self.kwargs.get('minval', 0.0)) / 2.0
            )

        else:
            return super().final_value()

    def parameter_value(self, *, step):
        if self.distribution == 'normal':
            parameter = tf.random.normal(
                shape=self.spec.shape, dtype=self.spec.tf_type(), mean=self.kwargs.get('mean', 0.0),
                stddev=self.kwargs.get('stddev', 1.0)
            )

        elif self.distribution == 'uniform':
            parameter = tf.random.uniform(
                shape=self.spec.shape, dtype=self.spec.tf_type(),
                minval=self.kwargs.get('minval', 0), maxval=self.kwargs.get('maxval')
            )

        return parameter
