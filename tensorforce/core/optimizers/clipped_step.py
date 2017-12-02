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


class ClippedStep(MetaOptimizer):
    """
    The multi-shep meta optimizer repeatedly applies the optimization step proposed by another  
    optimizer a number of times.
    """

    def __init__(self, optimizer, clip_delta_value=0.):
        """
        Creates a new multi-step meta optimizer instance.

        Args:
            optimizer: The optimizer which is modified by this meta optimizer.
            clip_delta_value: Clip deltas at this value.
        """
        super(ClippedStep, self).__init__(optimizer=optimizer)

        assert isinstance(clip_delta_value, float) and clip_delta_value > 0.
        self.clip_delta_value = clip_delta_value

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
        deltas = self.optimizer.step(time=time, variables=variables, **kwargs)

        with tf.control_dependencies(control_inputs=deltas):
            clipped_deltas = list()
            for var, delta in zip(variables, deltas):
                clipped_delta = tf.clip_by_value(delta, -self.clip_delta_value, self.clip_delta_value)
                var -= (delta - clipped_delta)
                clipped_deltas.append(clipped_delta)

        return clipped_deltas
