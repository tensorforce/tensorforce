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

from tensorforce import util
from tensorforce.core.optimizers import Optimizer


class Evolutionary(Optimizer):
    """Evolutionary optimizer."""

    def __init__(self, learning_rate, samples=1):
        super(Evolutionary, self).__init__()

        assert learning_rate > 0.0
        self.learning_rate = learning_rate

        assert samples >= 1
        self.samples = samples

    def tf_step(self, time, variables, fn_loss, **kwargs):
        """
        Evolutionary step function based on normal-sampled perturbations.

        :param fn_loss:

        :return:
        """
        unperturbed_loss = fn_loss()
        diffs = variables
        diffs_list = list()
        previous_perturbations = None

        for sample in xrange(self.samples):

            with tf.control_dependencies(control_inputs=diffs):
                perturbations = [tf.random_normal(shape=util.shape(t)) * self.learning_rate for t in diffs]
                if previous_perturbations is None:
                    applied = self.apply_step(variables=variables, diffs=perturbations)
                else:
                    perturbation_diffs = [pert - prev_pert for pert, prev_pert
                                          in zip(perturbations, previous_perturbations)]
                    applied = self.apply_step(variables=variables, diffs=perturbation_diffs)
                previous_perturbations = perturbations

            with tf.control_dependencies(control_inputs=(applied,)):
                perturbed_loss = fn_loss()
                direction = tf.sign(x=(unperturbed_loss - perturbed_loss))

            with tf.control_dependencies(control_inputs=(direction,)):
                diffs = [direction * perturbation for perturbation in perturbations]
                diffs_list.append(diffs)

        with tf.control_dependencies(control_inputs=diffs):
            diffs = [tf.add_n(inputs=[diffs[n] for diffs in diffs_list]) /
                     self.samples for n in range(len(diffs_list[0]))]
            perturbation_diffs = [diff - pert for diff, pert in zip(diffs, previous_perturbations)]
            applied = self.apply_step(variables=variables, diffs=perturbation_diffs)

        with tf.control_dependencies(control_inputs=(applied,)):
            return [diff + 0.0 for diff in diffs]
