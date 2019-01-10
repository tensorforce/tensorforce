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
        rmsprop=tf.train.RMSPropOptimizer
    )

    def __init__(self, name, optimizer, **kwargs):
        """
        Creates a new optimizer instance of a TensorFlow optimizer.

        Args:
            optimizer: The name of the optimizer. Must be a key of the tensorflow_optimizers dict.
            **kwargs: Arguments passed on to the TensorFlow optimizer constructor as **kwargs.
        """
        super().__init__(name=name)

        assert optimizer in TFOptimizer.tensorflow_optimizers
        self.optimizer = TFOptimizer.tensorflow_optimizers[optimizer](**kwargs)

    def tf_step(self, time, variables, arguments, fn_loss, **kwargs):
        """
        Keyword Args:
            arguments: Dict of arguments for passing to fn_loss as **kwargs.
            fn_loss: A callable taking arguments as kwargs and returning the loss op.
        """
        loss = fn_loss(**arguments)

        # Force loss value to be calculated.
        with tf.control_dependencies(control_inputs=(loss,)):
            # Trivial operation to enforce control dependency
            previous_variables = [util.identity_operation(x=variable) for variable in variables]

        # The actual tensorflow minimize op.
        with tf.control_dependencies(control_inputs=previous_variables):
            # colocate_gradients_with_ops=True
            applied = self.optimizer.minimize(loss=loss, var_list=variables)

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
