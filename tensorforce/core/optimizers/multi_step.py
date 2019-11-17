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
    Multi-step meta optimizer, which applies the given optimizer for a number of times
    (specification key: `multi_step`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        num_steps (parameter, int > 0): Number of optimization steps
            (<span style="color:#C00000"><b>required</b></span>).
        unroll_loop (bool): Whether to unroll the repetition loop
            (<span style="color:#00C000"><b>default</b></span>: false).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, optimizer, num_steps, unroll_loop=False, summary_labels=None):
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
        # Set reference to compare with at each optimization step, in case of a comparative loss.
        if fn_reference is not None:
            assert 'reference' not in arguments
            arguments['reference'] = fn_reference(**arguments)

        deltas = [tf.zeros_like(input=variable) for variable in variables]

        if self.unroll_loop:
            # Unrolled for loop
            for _ in range(self.num_steps):
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
                return (deltas,)

            num_steps = self.num_steps.value()
            deltas = self.while_loop(
                cond=util.tf_always_true, body=body, loop_vars=(deltas,), back_prop=False,
                maximum_iterations=num_steps
            )[0]

            return deltas
