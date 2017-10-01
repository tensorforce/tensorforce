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


class TFOptimizer(Optimizer):
    """
    Wrapper class for TensorFlow optimizers. This maps
    native TensorFlow optimizers to the TensorForce optimization interface.

    """

    tf_optimizers = dict(
        adadelta=tf.train.AdadeltaOptimizer,
        adagrad=tf.train.AdagradOptimizer,
        adam=tf.train.AdamOptimizer,
        gradient_descent=tf.train.GradientDescentOptimizer,
        momentum=tf.train.MomentumOptimizer,
        rmsprop=tf.train.RMSPropOptimizer
    )

    @staticmethod
    def get_wrapper(optimizer):
        def wrapper(**kwargs):
            return TFOptimizer(optimizer=optimizer, **kwargs)
        return wrapper

    def __init__(self, optimizer, **kwargs):
        super(TFOptimizer, self).__init__()

        self.optimizer = TFOptimizer.tf_optimizers[optimizer](**kwargs)

    def tf_step(self, time, variables, fn_loss, **kwargs):
        loss = fn_loss()

        with tf.control_dependencies(control_inputs=(loss,)):
            vars_before = [tf.add(x=var, y=0.0) for var in variables]

        with tf.control_dependencies(control_inputs=vars_before):
            applied = self.optimizer.minimize(loss=loss, var_list=variables)

        with tf.control_dependencies(control_inputs=(applied,)):
            return [var - var_before for var, var_before in zip(variables, vars_before)]
