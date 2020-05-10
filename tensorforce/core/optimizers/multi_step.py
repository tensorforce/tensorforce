# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce.core import parameter_modules, tf_function, tf_util
from tensorforce.core.optimizers import UpdateModifier


class MultiStep(UpdateModifier):
    """
    Multi-step update modifier, which applies the given optimizer for a number of times
    (specification key: `multi_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        num_steps (parameter, int >= 0): Number of optimization steps
            (<span style="color:#C00000"><b>required</b></span>).
        unroll_loop (bool): Whether to unroll the repetition loop
            (<span style="color:#00C000"><b>default</b></span>: false).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        optimized_module (module): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, optimizer, num_steps, unroll_loop=False, summary_labels=None, name=None,
        arguments_spec=None, optimized_module=None
    ):
        super().__init__(
            optimizer=optimizer, summary_labels=summary_labels, name=name,
            arguments_spec=arguments_spec, optimized_module=optimized_module
        )

        assert isinstance(unroll_loop, bool)
        self.unroll_loop = unroll_loop

        if self.unroll_loop:
            self.num_steps = num_steps
        else:
            self.num_steps = self.add_module(
                name='num_steps', module=num_steps, modules=parameter_modules, dtype='int',
                min_value=0
            )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        deltas = [tf.zeros_like(input=variable) for variable in variables]

        if self.unroll_loop:
            # Unrolled for loop
            for _ in range(self.num_steps):
                with tf.control_dependencies(control_inputs=deltas):
                    step_deltas = self.optimizer.step(
                        arguments=arguments, variables=variables, **kwargs
                    )
                    deltas = [delta1 + delta2 for delta1, delta2 in zip(deltas, step_deltas)]

            return deltas

        else:
            # TensorFlow while loop
            def body(deltas):
                with tf.control_dependencies(control_inputs=deltas):
                    step_deltas = self.optimizer.step(
                        arguments=arguments, variables=variables, **kwargs
                    )
                    deltas = [delta1 + delta2 for delta1, delta2 in zip(deltas, step_deltas)]
                return (deltas,)

            num_steps = self.num_steps.value()
            deltas = tf.while_loop(
                cond=tf_util.always_true, body=body, loop_vars=(deltas,),
                maximum_iterations=tf_util.int32(x=num_steps)
            )[0]

            return deltas
