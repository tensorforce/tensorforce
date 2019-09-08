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
from tensorforce.core.optimizers import MetaOptimizer


class GlobalOptimizer(MetaOptimizer):
    """
    Global meta optimizer, which applies the given optimizer to the local variables, then applies
    the update to a corresponding set of global variables, and subsequently updates the local
    variables to the value of the global variables; will likely change in the future (specification
    key: `global_optimizer`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def tf_step(self, variables, **kwargs):
        global_variables = kwargs["global_variables"]

        assert all(
            util.shape(global_variable) == util.shape(local_variable)
            for global_variable, local_variable in zip(global_variables, variables)
        )

        local_deltas = self.optimizer.step(variables=variables, **kwargs)

        with tf.control_dependencies(control_inputs=local_deltas):
            applied = self.optimizer.apply_step(variables=global_variables, deltas=local_deltas)

        with tf.control_dependencies(control_inputs=(applied,)):
            update_deltas = list()
            for global_variable, local_variable in zip(global_variables, variables):
                delta = global_variable - local_variable
                update_deltas.append(delta)

            applied = self.apply_step(variables=variables, deltas=update_deltas)

            # TODO: Update time, episode, etc (like in Synchronization)?

        with tf.control_dependencies(control_inputs=(applied,)):
            return [
                local_delta + update_delta
                for local_delta, update_delta in zip(local_deltas, update_deltas)
            ]
