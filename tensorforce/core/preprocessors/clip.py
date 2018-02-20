# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorforce.core.preprocessors import Preprocessor


class Clip(Preprocessor):
    """
    Clip by min/max.
    """

    def __init__(self, shape, min_value, max_value, scope='clip', summary_labels=()):
        self.min_value = min_value
        self.max_value = max_value
        super(Clip, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_process(self, tensor):
        return tf.clip_by_value(t=tensor, clip_value_min=self.min_value, clip_value_max=self.max_value)
