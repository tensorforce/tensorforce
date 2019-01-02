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
from tensorforce.core.layers import Layer


class Pooling(Layer):
    """
    Pooling layer (global pooling).
    """

    def __init__(self, name, pooling, input_spec):
        """
        Pooling constructor.

        Args:
            pooling ('concat' | 'max' | 'product' | 'sum'): Pooling type.
        """
        # Pooling
        if pooling not in ('concat', 'max', 'product', 'sum'):
            raise TensorforceError.value(name='pooling', argument='pooling', value=pooling)
        self.pooling = pooling

        super().__init__(name=name, input_spec=input_spec, l2_regularization=0.0)

    def default_input_spec(self):
        return dict(type='float', shape=None)

    def get_output_spec(self, input_spec):
        if self.pooling == 'concat':
            input_spec['shape'] = (util.product(xs=input_spec['shape']),)
        elif self.pooling == 'max':
            input_spec['shape'] = (input_spec['shape'][-1],)
        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_apply(self, x):
        if self.pooling == 'concat':
            return tf.reshape(tensor=x, shape=(-1, util.product(xs=util.shape(x)[1:])))

        elif self.pooling == 'max':
            for _ in range(util.rank(x=x) - 2):
                x = tf.reduce_max(input_tensor=x, axis=1)
            return x

        elif self.pooling == 'product':
            for _ in range(util.rank(x=x) - 2):
                x = tf.reduce_prod(input_tensor=x, axis=1)
            return x

        elif self.pooling == 'sum':
            for _ in range(util.rank(x=x) - 2):
                x = tf.reduce_sum(input_tensor=x, axis=1)
            return x
