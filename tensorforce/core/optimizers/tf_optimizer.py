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

    @staticmethod
    def get_wrapper(optimizer):
        """
        Returns a TFOptimizer constructor callable for the given optimizer name.

        Args:
            optimizer: The name of the optimizer, one of 'adadelta', 'adagrad', 'adam', 'nadam',  
            'gradient_descent', 'momentum', 'rmsprop'.

        Returns:
            The TFOptimizer constructor callable.
        """
        def wrapper(**kwargs):
            return TFOptimizer(optimizer=optimizer, **kwargs)
        return wrapper

    def __init__(self, optimizer, scope=None, summary_labels=(), **kwargs):
        """
        Creates a new optimizer instance of a TensorFlow optimizer.

        Args:
            optimizer: The name of the optimizer, one of 'adadelta', 'adagrad', 'adam', 'nadam',  
            'gradient_descent', 'momentum', 'rmsprop'.
            **kwargs: Additional arguments passed on to the TensorFlow optimizer constructor.
        """
        self.optimizer_spec = optimizer
        self.optimizer = TFOptimizer.tf_optimizers[optimizer](**kwargs)

        super(TFOptimizer, self).__init__(scope=(scope or optimizer), summary_labels=summary_labels)

    def tf_step(
        self,
        time,
        variables,
        arguments,
        fn_loss,
        **kwargs
    ):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            arguments: Dict of arguments for callables, like fn_loss.
            fn_loss: A callable returning the loss of the current model.
            **kwargs: Additional arguments, not used.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        loss = fn_loss(**arguments)

        with tf.control_dependencies(control_inputs=(loss,)):
            # Trivial operation to enforce control dependency
            previous_variables = [variable + 0.0 for variable in variables]

        with tf.control_dependencies(control_inputs=previous_variables):
            applied = self.optimizer.minimize(loss=loss, var_list=variables)  # colocate_gradients_with_ops=True

        with tf.control_dependencies(control_inputs=(applied,)):
            return [
                variable - previous_variable
                for variable, previous_variable in zip(variables, previous_variables)
            ]

    def get_variables(self):
        optimizer_variables = super(TFOptimizer, self).get_variables()

        slots_variables = [
            self.optimizer._slots[slot][key]
            for slot in sorted(self.optimizer._slots)
            for key in sorted(self.optimizer._slots[slot])
        ]

        if self.optimizer_spec in ('adam', 'nadam'):
            additional_variables = list(self.optimizer._get_beta_accumulators())
        else:
            additional_variables = list()

        return optimizer_variables + slots_variables + additional_variables
