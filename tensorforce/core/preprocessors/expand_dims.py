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

from tensorforce.core.preprocessors import Preprocessor


class ExpandDims(Preprocessor):
    """
    Expands dimensions of input tensor.
    """

    def __init__(self, shape, axis, scope='expand_dims', summary_labels=()):
        self.axis = axis
        super(ExpandDims, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def processed_shape(self, shape):
        position = self.axis if self.axis >= 0 else len(shape) + self.axis + 1
        return shape[:position] + (1,) + shape[position:]

    def tf_process(self, tensor):
        # Expand tensor.
        return tf.expand_dims(input=tensor, axis=self.axis)
