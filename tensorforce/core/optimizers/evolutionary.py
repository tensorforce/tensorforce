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
from tensorforce.core import parameter_modules
from tensorforce.core.optimizers import Optimizer


class Evolutionary(Optimizer):
    """
    Evolutionary optimizer, which samples random perturbations and applies them either as positive
    or negative update depending on their improvement of the loss (specification key:
    `evolutionary`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        learning_rate (parameter, float > 0.0): Learning rate
            (<span style="color:#C00000"><b>required</b></span>).
        num_samples (parameter, int > 0): Number of sampled perturbations
            (<span style="color:#00C000"><b>default</b></span>: 1).
        unroll_loop (bool): Whether to unroll the sampling loop
            (<span style="color:#00C000"><b>default</b></span>: false).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, learning_rate, num_samples=1, unroll_loop=False, summary_labels=None
    ):
        super().__init__(name=name, summary_labels=summary_labels)

        self.learning_rate = self.add_module(
            name='learning-rate', module=learning_rate, modules=parameter_modules, dtype='float'
        )

        assert isinstance(unroll_loop, bool)
        self.unroll_loop = unroll_loop

        if self.unroll_loop:
            self.num_samples = num_samples
        else:
            self.num_samples = self.add_module(
                name='num-samples', module=num_samples, modules=parameter_modules, dtype='int'
            )

    def tf_step(self, variables, arguments, fn_loss, **kwargs):
        learning_rate = self.learning_rate.value()
        unperturbed_loss = fn_loss(**arguments)

        deltas = [tf.zeros_like(input=variable) for variable in variables]
        previous_perturbations = [tf.zeros_like(input=variable) for variable in variables]

        if self.unroll_loop:
            # Unrolled for loop
            for sample in range(self.num_samples):
                with tf.control_dependencies(control_inputs=deltas):
                    perturbations = [
                        tf.random.normal(shape=util.shape(variable)) * learning_rate
                        for variable in variables
                    ]
                    perturbation_deltas = [
                        pert - prev_pert
                        for pert, prev_pert in zip(perturbations, previous_perturbations)
                    ]
                    applied = self.apply_step(variables=variables, deltas=perturbation_deltas)
                    previous_perturbations = perturbations

                with tf.control_dependencies(control_inputs=(applied,)):
                    perturbed_loss = fn_loss(**arguments)
                    direction = tf.sign(x=(unperturbed_loss - perturbed_loss))
                    deltas = [
                        delta + direction * perturbation
                        for delta, perturbation in zip(deltas, perturbations)
                    ]

        else:
            # TensorFlow while loop
            def body(deltas, previous_perturbations):
                with tf.control_dependencies(control_inputs=deltas):
                    perturbations = [
                        learning_rate * tf.random.normal(
                            shape=util.shape(x=variable), dtype=util.tf_dtype(dtype='float')
                        ) for variable in variables
                    ]
                    perturbation_deltas = [
                        pert - prev_pert
                        for pert, prev_pert in zip(perturbations, previous_perturbations)
                    ]
                    applied = self.apply_step(variables=variables, deltas=perturbation_deltas)

                with tf.control_dependencies(control_inputs=(applied,)):
                    perturbed_loss = fn_loss(**arguments)
                    direction = tf.sign(x=(unperturbed_loss - perturbed_loss))
                    deltas = [
                        delta + direction * perturbation
                        for delta, perturbation in zip(deltas, perturbations)
                    ]

                return deltas, perturbations

            num_samples = self.num_samples.value()
            deltas, perturbations = self.while_loop(
                cond=util.tf_always_true, body=body, loop_vars=(deltas, previous_perturbations),
                back_prop=False, maximum_iterations=num_samples
            )

        with tf.control_dependencies(control_inputs=deltas):
            num_samples = tf.dtypes.cast(x=num_samples, dtype=util.tf_dtype(dtype='float'))
            deltas = [delta / num_samples for delta in deltas]
            perturbation_deltas = [delta - pert for delta, pert in zip(deltas, perturbations)]
            applied = self.apply_step(variables=variables, deltas=perturbation_deltas)

        with tf.control_dependencies(control_inputs=(applied,)):
            # Trivial operation to enforce control dependency
            return util.fmap(function=util.identity_operation, xs=deltas)
