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
        learning_rate (parameter, float > 0.0): Learning rate
            (<span style="color:#C00000"><b>required</b></span>).
        num_samples (parameter, int >= 1): Number of sampled perturbations
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

        if num_samples is None:
            num_samples = 1
        self.num_samples = self.submodule(
            name='num_samples', module=num_samples, modules=parameter_modules, dtype='int',
            min_value=1
        )

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        learning_rate = self.learning_rate.value()

        unperturbed_loss = fn_loss(**arguments.to_kwargs())

        if self.num_samples.is_constant(value=1):
            deltas = list()
            for variable in variables:
                delta = tf.random.normal(shape=variable.shape, dtype=variable.dtype)
                if variable.dtype == tf_util.get_dtype(type='float'):
                    deltas.append(learning_rate * delta)
                else:
                    deltas.append(tf.cast(x=learning_rate, dtype=variable.dtype) * delta)

            assignments = list()
            for variable, delta in zip(variables, deltas):
                assignments.append(variable.assign_add(delta=delta, read_value=False))

            with tf.control_dependencies(control_inputs=assignments):
                perturbed_loss = fn_loss(**arguments.to_kwargs())

                def negate_deltas():
                    neg_two_float = tf_util.constant(value=-2.0, dtype='float')
                    assignments = list()
                    for variable, delta in zip(variables, deltas):
                        if variable.dtype == tf_util.get_dtype(type='float'):
                            assignments.append(
                                variable.assign_add(delta=(neg_two_float * delta), read_value=False)
                            )
                        else:
                            _ng_two_float = tf.constant(value=-2.0, dtype=variable.dtype)
                            assignments.append(
                                variable.assign_add(delta=(_ng_two_float * delta), read_value=False)
                            )

                    with tf.control_dependencies(control_inputs=assignments):
                        return [tf.math.negative(x=delta) for delta in deltas]

                return tf.cond(
                    pred=(perturbed_loss < unperturbed_loss), true_fn=(lambda: deltas),
                    false_fn=negate_deltas
                )

        else:
            deltas = [tf.zeros_like(input=variable) for variable in variables]
            previous_perturbations = [tf.zeros_like(input=variable) for variable in variables]

            def body(deltas, previous_perturbations):
                with tf.control_dependencies(control_inputs=deltas):
                    perturbations = list()
                    for variable in variables:
                        perturbation = tf.random.normal(shape=variable.shape, dtype=variable.dtype)
                        if variable.dtype == tf_util.get_dtype(type='float'):
                            perturbations.append(learning_rate * perturbation)
                        else:
                            perturbations.append(
                                tf.cast(x=learning_rate, dtype=variable.dtype) * perturbation
                            )

                    perturbation_deltas = [
                        pert - prev_pert
                        for pert, prev_pert in zip(perturbations, previous_perturbations)
                    ]
                    assignments = list()
                    for variable, delta in zip(variables, perturbation_deltas):
                        assignments.append(variable.assign_add(delta=delta, read_value=False))

                with tf.control_dependencies(control_inputs=assignments):
                    perturbed_loss = fn_loss(**arguments.to_kwargs())

                    one_float = tf_util.constant(value=1.0, dtype='float')
                    neg_one_float = tf_util.constant(value=-1.0, dtype='float')
                    direction = tf.where(
                        condition=(perturbed_loss < unperturbed_loss), x=one_float, y=neg_one_float
                    )

                    next_deltas = list()
                    for variable, delta, perturbation in zip(variables, deltas, perturbations):
                        if variable.dtype == tf_util.get_dtype(type='float'):
                            next_deltas.append(delta + direction * perturbation)
                        else:
                            next_deltas.append(
                                delta + tf.cast(x=direction, dtype=variable.dtype) * perturbation
                            )

                return next_deltas, perturbations

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
