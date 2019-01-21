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
    Evolutionary optimizer which samples random perturbations and applies them either positively  
    or negatively, depending on their improvement of the loss.
    """

    def __init__(self, name, learning_rate, num_samples=1, unroll_loop=False, summary_labels=None):
        """
        Creates a new evolutionary optimizer instance.

        Args:
            learning_rate: Learning rate.
            num_samples: Number of sampled perturbations.
        """
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
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            variables: List of variables to optimize.
            arguments: Dict of arguments for callables, like fn_loss.
            fn_loss: A callable returning the loss of the current model.
            **kwargs: Additional arguments, not used.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        unperturbed_loss = fn_loss(**arguments)

        # First sample
        learning_rate = self.learning_rate.value()
        perturbations = [
            tf.random_normal(shape=util.shape(variable)) * learning_rate
            for variable in variables
        ]
        applied = self.apply_step(variables=variables, deltas=perturbations)

        with tf.control_dependencies(control_inputs=(applied,)):
            perturbed_loss = fn_loss(**arguments)
            direction = tf.sign(x=(unperturbed_loss - perturbed_loss))
            deltas_sum = [direction * perturbation for perturbation in perturbations]

        if self.unroll_loop:
            # Unrolled for loop
            previous_perturbations = perturbations
            for sample in range(self.num_samples - 1):

                with tf.control_dependencies(control_inputs=deltas_sum):
                    perturbations = [
                        tf.random_normal(shape=util.shape(variable)) * learning_rate
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
                    deltas_sum = [
                        delta + direction * perturbation
                        for delta, perturbation in zip(deltas_sum, perturbations)
                    ]

        else:
            # TensorFlow while loop
            def body(deltas_sum, previous_perturbations):

                with tf.control_dependencies(control_inputs=deltas_sum):
                    perturbations = [
                        tf.random_normal(shape=util.shape(variable)) * learning_rate
                        for variable in variables
                    ]
                    perturbation_deltas = [
                        pert - prev_pert
                        for pert, prev_pert in zip(perturbations, previous_perturbations)
                    ]
                    applied = self.apply_step(variables=variables, deltas=perturbation_deltas)

                with tf.control_dependencies(control_inputs=(applied,)):
                    perturbed_loss = fn_loss(**arguments)
                    direction = tf.sign(x=(unperturbed_loss - perturbed_loss))
                    deltas_sum = [
                        delta + direction * perturbation
                        for delta, perturbation in zip(deltas_sum, perturbations)
                    ]

                return deltas_sum, perturbations

            num_samples = self.num_samples.value()
            one = tf.constant(value=1, dtype=util.tf_dtype(dtype='int'))
            deltas_sum, perturbations = self.while_loop(
                cond=util.tf_always_true, body=body, loop_vars=(deltas_sum, perturbations),
                maximum_iterations=(num_samples - one)
            )

        with tf.control_dependencies(control_inputs=deltas_sum):
            num_samples = tf.dtypes.cast(x=num_samples, dtype=util.tf_dtype(dtype='float'))
            deltas = [delta / num_samples for delta in deltas_sum]
            perturbation_deltas = [delta - pert for delta, pert in zip(deltas, perturbations)]
            applied = self.apply_step(variables=variables, deltas=perturbation_deltas)

        with tf.control_dependencies(control_inputs=(applied,)):
            # Trivial operation to enforce control dependency
            return [util.identity_operation(x=delta) for delta in deltas]
