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
        num_steps (parameter, int >= 1): Number of optimization steps
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, optimizer, num_steps, name=None, arguments_spec=None):
        super().__init__(optimizer=optimizer, name=name, arguments_spec=arguments_spec)

        self.num_steps = self.submodule(
            name='num_steps', module=num_steps, modules=parameter_modules, dtype='int',
            min_value=1
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        deltas = [tf.zeros_like(input=variable) for variable in variables]

        def body(*deltas):
            with tf.control_dependencies(control_inputs=deltas):
                step_deltas = self.optimizer.step(
                    arguments=arguments, variables=variables, **kwargs
                )
                deltas = [delta1 + delta2 for delta1, delta2 in zip(deltas, step_deltas)]
            return deltas

        num_steps = self.num_steps.value()
        deltas = tf.while_loop(
            cond=tf_util.always_true, body=body, loop_vars=deltas,
            maximum_iterations=tf_util.int32(x=num_steps)
        )

        return deltas
