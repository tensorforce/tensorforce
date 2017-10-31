# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce import util
from tensorforce.core.optimizers import MetaOptimizer


class GlobalOptimizer(MetaOptimizer):
    """
    The global optimizer applies an optimizer to the local variables. In addition, it also  
    applies the update a corresponding set of global variables and subsequently updates the local  
    variables to the value of these global variables.
    Note: This is used for the current distributed mode, and will likely change with the next  
    major version update.
    """

    def __init__(self, optimizer):
        """
        Creates a new global optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
        """
        super(GlobalOptimizer, self).__init__(optimizer=optimizer)

    def tf_step(self, time, variables, global_variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            global_variables: List of global variables to apply the proposed optimization step to.
            **kwargs: ??? coming soon

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        assert all(util.shape(global_var) == util.shape(local_var) for global_var, local_var in zip(global_variables, variables))

        local_deltas = self.optimizer.step(time=time, variables=variables, **kwargs)

        with tf.control_dependencies(control_inputs=local_deltas):
            applied = self.optimizer.apply_step(variables=global_variables, deltas=local_deltas)

        with tf.control_dependencies(control_inputs=(applied,)):
            update_deltas = list()
            for global_var, local_var in zip(global_variables, variables):
                delta = global_var - local_var
                update_deltas.append(delta)

            applied = self.apply_step(variables=variables, deltas=update_deltas)

            # TODO: Update time, episode, etc (like in Synchronization)?

        with tf.control_dependencies(control_inputs=(applied,)):
            return [local_delta + update_delta for local_delta, update_delta in zip(local_deltas, update_deltas)]
