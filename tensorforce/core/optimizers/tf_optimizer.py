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
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce.core.optimizers import Optimizer


class TensorFlowOptimizer(Optimizer):

    tf_optimizers = dict(
        adadelta=tf.train.AdadeltaOptimizer,
        adagrad=tf.train.AdagradOptimizer,
        adam=tf.train.AdamOptimizer,
        gradient_descent=tf.train.GradientDescentOptimizer,
        momentum=tf.train.MomentumOptimizer,
        rmsprop=tf.train.RMSPropOptimizer
    )

    @classmethod
    def get_wrapper(cls, optimizer):
        def wrapper(**kwargs):
            return cls(optimizer=optimizer, **kwargs)
        return wrapper

    def __init__(self, optimizer, **kwargs):
        super(TensorFlowOptimizer, self).__init__()
        self.optimizer = TensorFlowOptimizer.tf_optimizers[optimizer](**kwargs)

    def minimize(self, fn_loss, fn_kl_divergence=None, variables=None):
        variables = self.get_variables(variables=variables)
        if isinstance(fn_loss, tf.Tensor):  # TEMPORARY !!!!!!!!
            return self.optimizer.minimize(loss=fn_loss, var_list=variables)
        else:
            return self.optimizer.minimize(loss=fn_loss(), var_list=variables)
