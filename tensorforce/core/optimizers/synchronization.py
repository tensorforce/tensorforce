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
from tensorforce.core.optimizers import Optimizer


class Synchronization(Optimizer):
    """
    Synchronization optimizer updating variables periodically to the value of a set of source variables.
    """

    def __init__(self, update_frequency=1, update_weight=1.0):
        super(Synchronization, self).__init__()

        assert isinstance(update_frequency, int) and update_frequency > 0
        self.update_frequency = update_frequency

        assert isinstance(update_weight, float) and update_weight > 0.0
        self.update_weight = update_weight

    def tf_step(self, time, variables, source_variables, **kwargs):
        last_update = tf.get_variable(name='last-update', dtype=tf.int32, initializer=(-self.update_frequency), trainable=False)

        def true_fn():
            diffs = list()
            for source, target in zip(source_variables, variables):
                diff = self.update_weight * (source - target)
                diffs.append(diff)

            applied = self.apply_step(variables=variables, diffs=diffs)
            update_time = last_update.assign(value=time)

            with tf.control_dependencies(control_inputs=(applied, update_time)):
                return [tf.identity(input=diff) for diff in diffs]

        def false_fn():
            diffs = list()
            for variable in variables:
                diff = tf.zeros(shape=util.shape(variable))
                diffs.append(diff)
            return diffs

        do_sync = (time - last_update >= self.update_frequency)
        return tf.cond(pred=do_sync, true_fn=true_fn, false_fn=false_fn)
