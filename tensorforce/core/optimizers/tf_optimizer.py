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
    Wrapper class for TensorFlow optimizers.
    """

    tf_optimizers = dict(
        adadelta=tf.train.AdadeltaOptimizer,
        adagrad=tf.train.AdagradOptimizer,
        adam=tf.train.AdamOptimizer,
        nadam=tf.contrib.opt.NadamOptimizer,
        gradient_descent=tf.train.GradientDescentOptimizer,
        momentum=tf.train.MomentumOptimizer,
        rmsprop=tf.train.RMSPropOptimizer
    )

    def __init__(self, optimizer, scope=None, summary_labels=(), **kwargs):
        """
        Creates a new optimizer instance of a TensorFlow optimizer.

        Args:
            optimizer: The name of the optimizer. Must be one of the keys of the tf_optimizers dict.
            **kwargs: Arguments passed on to the TensorFlow optimizer constructor as **kwargs.
        """
        self.tf_optimizer_type = optimizer
        self.tf_optimizer = TFOptimizer.tf_optimizers[optimizer](**kwargs)

        super(TFOptimizer, self).__init__(scope=(scope or optimizer), summary_labels=summary_labels)

    def tf_step(self, time, variables, **kwargs):
        """
        Keyword Args:
            arguments: Dict of arguments for passing to fn_loss as **kwargs.
            fn_loss: A callable taking arguments as kwargs and returning the loss op of the current model.
        """
        arguments = kwargs["arguments"]
        fn_loss = kwargs["fn_loss"]
        loss = fn_loss(**arguments)

        # Force loss value to be calculated.
        with tf.control_dependencies(control_inputs=(loss,)):
            # Trivial operation to enforce control dependency
            previous_variables = [variable + 0.0 for variable in variables]

        # The actual tensorflow minimize op.
        with tf.control_dependencies(control_inputs=previous_variables):
            applied = self.tf_optimizer.minimize(loss=loss, var_list=variables)  # colocate_gradients_with_ops=True

        # Return deltas after actually having change the variables.
        with tf.control_dependencies(control_inputs=(applied,)):
            return [
                variable - previous_variable
                for variable, previous_variable in zip(variables, previous_variables)
            ]

    def get_variables(self):
        optimizer_variables = super(TFOptimizer, self).get_variables()

        slots_variables = [
            self.tf_optimizer._slots[slot][key]
            for slot in sorted(self.tf_optimizer._slots)
            for key in sorted(self.tf_optimizer._slots[slot])
        ]

        if self.tf_optimizer_type in ('adam', 'nadam'):
            additional_variables = list(self.tf_optimizer._get_beta_accumulators())
        else:
            additional_variables = list()

        return optimizer_variables + slots_variables + additional_variables
