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
        deltas = variables
        deltas_list = list()
        previous_perturbations = None

        for sample in xrange(self.samples):

            with tf.control_dependencies(control_inputs=deltas):
                perturbations = [tf.random_normal(shape=util.shape(t)) * self.learning_rate for t in deltas]
                if previous_perturbations is None:
                    applied = self.apply_step(variables=variables, deltas=perturbations)
                else:
                    perturbation_deltas = [pert - prev_pert for pert, prev_pert
                                          in zip(perturbations, previous_perturbations)]
                    applied = self.apply_step(variables=variables, deltas=perturbation_deltas)
                previous_perturbations = perturbations

            with tf.control_dependencies(control_inputs=(applied,)):
                perturbed_loss = fn_loss()
                direction = tf.sign(x=(unperturbed_loss - perturbed_loss))

            with tf.control_dependencies(control_inputs=(direction,)):
                deltas = [direction * perturbation for perturbation in perturbations]
                deltas_list.append(deltas)

        with tf.control_dependencies(control_inputs=deltas):
            deltas = [tf.add_n(inputs=[deltas[n] for deltas in deltas_list]) /
                     self.samples for n in range(len(deltas_list[0]))]
            perturbation_deltas = [delta - pert for delta, pert in zip(deltas, previous_perturbations)]
            applied = self.apply_step(variables=variables, deltas=perturbation_deltas)

        with tf.control_dependencies(control_inputs=(applied,)):
            return [delta + 0.0 for delta in deltas]
