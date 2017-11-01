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

from six.moves import xrange
import tensorflow as tf

from tensorforce.core.optimizers import MetaOptimizer


class MultiStep(MetaOptimizer):
    """
    The multi-shep meta optimizer repeatedly applies the optimization step proposed by another  
    optimizer a number of times.
    """

    def __init__(self, optimizer, num_steps=5):
        """
        Creates a new multi-step meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
            num_steps: Number of optimization steps to perform.
        """
        super(MultiStep, self).__init__(optimizer=optimizer)

        assert isinstance(num_steps, int) and num_steps > 0
        self.num_steps = num_steps

    def tf_step(self, time, variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            **kwargs: Additional arguments passed on to the internal optimizer.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        overall_deltas = None
        deltas = ()
        for _ in xrange(self.num_steps):

            with tf.control_dependencies(control_inputs=deltas):
                deltas = self.optimizer.step(time=time, variables=variables, **kwargs)

                if overall_deltas is None:
                    overall_deltas = deltas
                else:
                    overall_deltas = [delta1 + delta2 for delta1, delta2 in zip(overall_deltas, deltas)]

        return overall_deltas
