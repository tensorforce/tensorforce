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


class Grayscale(Preprocessor):
    """
    Turn 3D color state into grayscale.
    """

    def __init__(self, shape, weights=(0.299, 0.587, 0.114), remove_rank=False, scope='grayscale', summary_labels=()):
        """
        Args:
            weights (tuple): The weights to multiply each color channel with (in order: red, blue, green).
            remove_rank (bool): If True, will remove the color channel rank from the input tensor.
        """
        self.weights = weights
        self.remove_rank = remove_rank
        super(Grayscale, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_process(self, tensor):
        weights = tf.reshape(tensor=self.weights, shape=(tuple(1 for _ in range(util.rank(tensor) - 1)) + (3,)))
        weighted_sum = tf.reduce_sum(input_tensor=(weights * tensor), axis=-1, keepdims=(not self.remove_rank))
        return weighted_sum

    def processed_shape(self, shape):
        return tuple(shape[:-1]) + ((1,) if not self.remove_rank else ())
