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

from tensorforce.core import tf_function, tf_util
from tensorforce.core.optimizers import UpdateModifier


class GlobalOptimizer(UpdateModifier):
    """
    Global update modifier, which applies the given optimizer to the local variables, then applies
    the update to a corresponding set of global variables, and subsequently updates the local
    variables to the value of the global variables; will likely change in the future (specification
    key: `global_optimizer`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        global_variables = kwargs['global_variables']

        assert all(
            tf_util.shape(x=global_variable) == tf_util.shape(x=local_variable)
            for global_variable, local_variable in zip(global_variables, variables)
        )

        local_deltas = self.optimizer.step(arguments=arguments, variables=variables, **kwargs)

        with tf.control_dependencies(control_inputs=local_deltas):
            assignments = list()
            for variable, delta in zip(global_variables, local_deltas):
                assignments.append(variable.assign_add(delta=delta, read_value=False))

        with tf.control_dependencies(control_inputs=assignments):
            update_deltas = list()
            for global_variable, local_variable in zip(global_variables, variables):
                update_deltas.append(global_variable - local_variable)

            assignments = list()
            for variable, delta in zip(variables, update_deltas):
                assignments.append(variable.assign_add(delta=delta, read_value=False))

            # TODO: Update time, episode, etc (like in Synchronization)?

        with tf.control_dependencies(control_inputs=assignments):
            return [
                local_delta + update_delta
                for local_delta, update_delta in zip(local_deltas, update_deltas)
            ]
