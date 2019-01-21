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
from tensorforce.core.optimizers import MetaOptimizer


class MultiStep(MetaOptimizer):
    """
    The multi-step meta optimizer repeatedly applies the optimization step proposed by another  
    optimizer a number of times.
    """

    def __init__(self, name, optimizer, num_steps=10, unroll_loop=False, summary_labels=None):
        """
        Creates a new multi-step meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
            num_steps: Number of optimization steps to perform.
        """
        super().__init__(name=name, optimizer=optimizer, summary_labels=summary_labels)

        assert isinstance(unroll_loop, bool)
        self.unroll_loop = unroll_loop

        if self.unroll_loop:
            self.num_steps = num_steps
        else:
            self.num_steps = self.add_module(
                name='num-steps', module=num_steps, modules=parameter_modules, dtype='int'
            )

    def tf_step(self, variables, arguments, fn_reference=None, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            variables: List of variables to optimize.
            arguments: Dict of arguments for callables, like fn_loss.
            fn_reference: A callable returning the reference values, in case of a comparative loss.
            **kwargs: Additional arguments passed on to the internal optimizer.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """

        # Set reference to compare with at each optimization step, in case of a comparative loss.
        arguments['reference'] = fn_reference(**arguments)

        # First step
        deltas = self.optimizer.step(variables=variables, arguments=arguments, **kwargs)

        if self.unroll_loop:
            # Unrolled for loop
            for _ in range(self.num_steps - 1):
                with tf.control_dependencies(control_inputs=deltas):
                    step_deltas = self.optimizer.step(
                        variables=variables, arguments=arguments, **kwargs
                    )
                    deltas = [delta1 + delta2 for delta1, delta2 in zip(deltas, step_deltas)]

            return deltas

        else:
            # TensorFlow while loop
            def body(deltas):
                with tf.control_dependencies(control_inputs=deltas):
                    step_deltas = self.optimizer.step(
                        variables=variables, arguments=arguments, **kwargs
                    )
                    deltas = [delta1 + delta2 for delta1, delta2 in zip(deltas, step_deltas)]
                    return deltas

            num_steps = self.num_steps.value()
            one = tf.constant(value=1, dtype=util.tf_dtype(dtype='int'))
            deltas = self.while_loop(
                cond=util.tf_always_true, body=body, loop_vars=(deltas,),
                maximum_iterations=(num_steps - one)
            )

            return deltas
