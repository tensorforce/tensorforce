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
    """

    def __init__(self, name, dtype, distribution, shape=(), summary_labels=None, **kwargs):
        super().__init__(name=name, dtype=dtype, shape=shape, summary_labels=summary_labels)

        assert distribution in ('normal', 'uniform')

        self.distribution = distribution
        self.kwargs = kwargs

    def get_parameter_value(self):
        if self.distribution == 'normal':
            parameter = tf.random.normal(
                shape=self.shape, dtype=util.tf_dtype(dtype=self.dtype), **self.kwargs
            )

        elif self.distribution == 'uniform':
            parameter = tf.random.uniform(
                shape=self.shape, dtype=util.tf_dtype(dtype=self.dtype), **self.kwargs
            )

        return parameter
