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
from tensorforce.core import Module


class Optimizer(Module):
    """
    Base class for optimizers which minimize a not yet further specified expression, usually some
    kind of loss function. More generally, an optimizer can be considered as some method of
    updating a set of variables.
    """

    def __init__(self, name, summary_labels=None):
        super().__init__(name=name, l2_regularization=0.0, summary_labels=summary_labels)

    def tf_step(self, variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step on the given  
        variables, including actually changing the values of the variables.

        Args:
            variables: List of variables to optimize.
            **kwargs: Additional arguments depending on the specific optimizer implementation.  
                For instance, often includes `fn_loss` if a loss function is optimized.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        raise NotImplementedError

    def tf_apply_step(self, variables, deltas):
        """
        Applies the given (and already calculated) step deltas to the variable values.

        Args:
            variables: List of variables.
            deltas: List of deltas of same length.

        Returns:
            The step-applied operation. A tf.group of tf.assign_add ops.
        """
        if len(variables) != len(deltas):
            raise TensorforceError("Invalid variables and deltas lists.")

        return tf.group(*(
            tf.assign_add(ref=variable, value=delta) for variable, delta in zip(variables, deltas)
        ))

    def tf_minimize(self, variables, **kwargs):
        """
        Performs an optimization step.

        Args:
            variables: List of variables to optimize.
            **kwargs: Additional optimizer-specific arguments. The following arguments are used
                by some optimizers:
            - arguments: Dict of arguments for callables, like fn_loss.
            - fn_loss: A callable returning the loss of the current model.
            - fn_reference: A callable returning the reference values, in case of a comparative  
                loss.
            - fn_kl_divergence: A callable returning the KL-divergence relative to the
                current model.
            - sampled_loss: A sampled loss (integer).
            - return_estimated_improvement: Returns the estimated improvement resulting from
                the natural gradient calculation if true.
            - source_variables: List of source variables to synchronize with.
            - global_variables: List of global variables to apply the proposed optimization
                step to.


        Returns:
            The optimization operation.
        """
        deltas = self.step(variables=variables, **kwargs)

        for n in range(len(variables)):
            name = variables[n].name
            if name[-2:] != ':0':
                raise TensorforceError.unexpected()
            deltas[n] = self.add_summary(
                label='updates', name=(name[:-2] + '-update'), tensor=deltas[n], mean_variance=True
            )

        with tf.control_dependencies(control_inputs=deltas):
            return util.no_operation()

    def add_variable(self, name, dtype, shape, is_trainable=False, initializer='zeros'):
        if is_trainable:
            raise TensorforceError("Invalid trainable variable.")

        return super().add_variable(
            name=name, dtype=dtype, shape=shape, is_trainable=is_trainable, initializer=initializer
        )
