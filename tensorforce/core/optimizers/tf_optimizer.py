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
from tensorforce.core import parameter_modules
from tensorforce.core.optimizers import Optimizer


class TFOptimizer(Optimizer):
    """
    Wrapper class for TensorFlow optimizers.
    """

    tensorflow_optimizers = dict(
        adadelta=tf.train.AdadeltaOptimizer,
        adagrad=tf.train.AdagradOptimizer,
        adam=tf.train.AdamOptimizer,
        gradient_descent=tf.train.GradientDescentOptimizer,
        momentum=tf.train.MomentumOptimizer,
        nadam=tf.contrib.opt.NadamOptimizer,
        proximal_adagrad=tf.train.ProximalAdagradOptimizer,
        proximal_gradient_descent=tf.train.ProximalGradientDescentOptimizer,
        rmsprop=tf.train.RMSPropOptimizer
    )

    def __init__(self, name, optimizer, learning_rate, summary_labels=None, **kwargs):
        """
        Creates a new optimizer instance of a TensorFlow optimizer.

        Args:
            optimizer: The name of the optimizer. Must be a key of the tensorflow_optimizers dict.
            **kwargs: Arguments passed on to the TensorFlow optimizer constructor as **kwargs.
        """
        super().__init__(name=name, summary_labels=summary_labels)

        assert optimizer in TFOptimizer.tensorflow_optimizers
        self.learning_rate = self.add_module(
            name='learning-rate', module=learning_rate, modules=parameter_modules, dtype='float'
        )
        self.optimizer = TFOptimizer.tensorflow_optimizers[optimizer]
        self.optimizer_kwargs = kwargs

    def tf_initialize(self):
        super().tf_initialize()

        self.optimizer = self.optimizer(
            learning_rate=self.learning_rate.value, **self.optimizer_kwargs
        )

    def tf_step(self, variables, arguments, fn_loss, **kwargs):
        """
        Keyword Args:
            arguments: Dict of arguments for passing to fn_loss as **kwargs.
            fn_loss: A callable taking arguments as kwargs and returning the loss op.
        """
        # Trivial operation to enforce control dependency
        previous_variables = [util.identity_operation(x=variable) for variable in variables]

        # Force loss value to be calculated.
        with tf.control_dependencies(control_inputs=previous_variables):
            loss = fn_loss(**arguments)

        # The actual tensorflow minimize op.
        applied = self.optimizer.minimize(loss=loss, var_list=variables)
        # colocate_gradients_with_ops=True

        # Return deltas after actually having change the variables.
        with tf.control_dependencies(control_inputs=(applied,)):
            return [
                variable - previous_variable
                for variable, previous_variable in zip(variables, previous_variables)
            ]

    def get_variables(self, only_trainable=True):
        variables = super().get_variables(only_trainable=only_trainable)

        variables.extend(self.optimizer.variables())

        # variables.extend(
        #     self.optimizer._slots[slot][key] for slot in sorted(self.optimizer._slots)
        #     for key in sorted(self.optimizer._slots[slot])
        # )

        # if isinstance(self.optimizer, (tf.train.AdamOptimizer, tf.contrib.opt.NadamOptimizer)):
        #     variables.extend(self.optimizer._get_beta_accumulators())

        return variables
