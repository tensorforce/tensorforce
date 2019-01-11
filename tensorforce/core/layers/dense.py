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

from tensorforce.core.layers import TransformationBase


class Dense(TransformationBase):
    """
    Dense fully-connected layer.
    """

    def __init__(
        self, name, size, bias=True, activation='relu', dropout=None, input_spec=None,
        l2_regularization=None, summary_labels=None
    ):
        super().__init__(
            name=name, size=size, bias=bias, activation=activation, dropout=dropout,
            input_spec=input_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

    def default_input_spec(self):
        return dict(type='float', shape=(0,))

    def get_output_spec(self, input_spec):
        if self.squeeze:
            input_spec['shape'] = input_spec['shape'][:-1]
        else:
            input_spec['shape'] = input_spec['shape'][:-1] + (self.size,)
        input_spec.pop('min_value', None)
        input_spec.pop('max_value', None)

        return input_spec

    def tf_initialize(self):
        super().tf_initialize()

        in_size = self.input_spec['shape'][0]
        self.weights = self.add_variable(
            name='weights', dtype='float', shape=(in_size, self.size), is_trainable=True,
            initializer='random'
        )

    def tf_apply(self, x):
        # tf.assert_rank_in(x=x, ranks=(2, 3, 4))
        x = tf.matmul(a=x, b=self.weights)

        return super().tf_apply(x=x)


class Linear(Dense):
    """
    Linear layer.
    """

    def __init__(self, name, size, bias=True, input_spec=None, summary_labels=None):
        """
        Linear constructor.
        """
        super().__init__(
            name=name, size=size, bias=bias, activation=None, dropout=None, input_spec=input_spec,
            l2_regularization=0.0, summary_labels=summary_labels
        )
