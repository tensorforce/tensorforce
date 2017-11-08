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

import numpy as np
import scipy

from tensorforce.core.preprocessing import Preprocessor


class ImageResize(Preprocessor):
    """
    Resize image to width x height.
    """

    def __init__(self, width, height):
        super(ImageResize, self).__init__()
        self.size = (width, height)

    def process(self, state):
        return scipy.misc.imresize(arr=state.astype(np.uint8), size=self.size)

    def processed_shape(self, shape):
        return self.size + (shape[-1],)
