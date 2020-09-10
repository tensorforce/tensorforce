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

from tensorforce.core import tf_function
from tensorforce.core.optimizers import UpdateModifier


class DoublecheckStep(UpdateModifier):
    """
    Double-check update modifier, which checks whether the update of the given optimizer has
    decreased the loss and otherwise reverses it (specification key: `doublecheck_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        loss_before = fn_loss(**arguments.to_kwargs())

        with tf.control_dependencies(control_inputs=(loss_before,)):
            deltas = self.optimizer.step(
                arguments=arguments, variables=variables, fn_loss=fn_loss, **kwargs
            )

        with tf.control_dependencies(control_inputs=deltas):
            loss_after = fn_loss(**arguments.to_kwargs())

            def reverse_update():
                assignments = list()
                for variable, delta in zip(variables, deltas):
                    assignments.append(variable.assign_add(delta=-delta, read_value=False))
                with tf.control_dependencies(control_inputs=assignments):
                    return [tf.zeros_like(input=delta) for delta in deltas]

            is_improvement = (loss_after < loss_before)
            return tf.cond(pred=is_improvement, true_fn=(lambda: deltas), false_fn=reverse_update)
