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
from tensorforce.core.preprocessors import Preprocessor


class Normalize(Preprocessor):
    """
    Normalize state. Subtract minimal value and divide by range.
    """

    def __init__(self, shape, scope='normalize', summary_labels=()):
        super(Normalize, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_process(self, tensor):
        # Min/max across every axis except batch dimension.
        min_value = tensor
        max_value = tensor
        for axis in range(1, util.rank(tensor)):
            min_value = tf.reduce_min(input_tensor=min_value, axis=axis, keep_dims=True)
            max_value = tf.reduce_max(input_tensor=max_value, axis=axis, keep_dims=True)

        return (tensor - min_value) / (max_value - min_value + util.epsilon)
