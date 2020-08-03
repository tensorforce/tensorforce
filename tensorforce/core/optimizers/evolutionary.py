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

from tensorforce.core import parameter_modules, tf_function, tf_util
from tensorforce.core.optimizers import Optimizer


class Evolutionary(Optimizer):
    """
    Evolutionary optimizer, which samples random perturbations and applies them either as positive
    or negative update depending on their improvement of the loss (specification key:
    `evolutionary`).

    Args:
        learning_rate (parameter, float >= 0.0): Learning rate
            (<span style="color:#C00000"><b>required</b></span>).
        num_samples (parameter, int >= 0): Number of sampled perturbations
            (<span style="color:#00C000"><b>default</b></span>: 1).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, learning_rate, num_samples=1, name=None, arguments_spec=None):
        super().__init__(name=name, arguments_spec=arguments_spec)

        self.learning_rate = self.submodule(
            name='learning_rate', module=learning_rate, modules=parameter_modules, dtype='float',
            min_value=0.0
        )

        self.num_samples = self.submodule(
            name='num_samples', module=num_samples, modules=parameter_modules, dtype='int',
            min_value=1
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        learning_rate = self.learning_rate.value()

        unperturbed_loss = fn_loss(**arguments.to_kwargs())

        deltas = [tf.zeros_like(input=variable) for variable in variables]
        previous_perturbations = [tf.zeros_like(input=variable) for variable in variables]

        def body(deltas, previous_perturbations):
            with tf.control_dependencies(control_inputs=deltas):
                perturbations = [
                    learning_rate * tf.random.normal(
                        shape=tf_util.shape(x=variable), dtype=tf_util.get_dtype(type='float')
                    ) for variable in variables
                ]
                perturbation_deltas = [
                    pert - prev_pert
                    for pert, prev_pert in zip(perturbations, previous_perturbations)
                ]
                assignments = list()
                for variable, delta in zip(variables, perturbation_deltas):
                    assignments.append(variable.assign_add(delta=delta, read_value=False))

            with tf.control_dependencies(control_inputs=assignments):
                perturbed_loss = fn_loss(**arguments.to_kwargs())
                direction = tf.math.sign(x=(unperturbed_loss - perturbed_loss))
                deltas = [
                    delta + direction * perturbation
                    for delta, perturbation in zip(deltas, perturbations)
                ]

            return deltas, perturbations

        num_samples = self.num_samples.value()
        deltas, perturbations = tf.while_loop(
            cond=tf_util.always_true, body=body, loop_vars=(deltas, previous_perturbations),
            maximum_iterations=tf_util.int32(x=num_samples)
        )

        with tf.control_dependencies(control_inputs=deltas):
            num_samples = tf_util.cast(x=num_samples, dtype='float')
            deltas = [delta / num_samples for delta in deltas]
            perturbation_deltas = [delta - pert for delta, pert in zip(deltas, perturbations)]
            assignments = list()
            for variable, delta in zip(variables, perturbation_deltas):
                assignments.append(variable.assign_add(delta=delta, read_value=False))

        with tf.control_dependencies(control_inputs=assignments):
            # Trivial operation to enforce control dependency
            return [tf_util.identity(input=delta) for delta in deltas]
