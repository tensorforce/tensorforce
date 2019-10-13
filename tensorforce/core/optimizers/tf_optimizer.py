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


class TFOptimizer(Optimizer):
    """
    TensorFlow optimizer (specification key: `tf_optimizer`, `adadelta`, `adagrad`, `adam`,
    `gradient_descent`, `momentum`, `proximal_adagrad`, `proximal_gradient_descent`, `rmsprop`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer ('adadelta' | 'adagrad' | 'adam' | 'gradient_descent' | 'momentum' | 'proximal_adagrad' | 'proximal_gradient_descent' | 'rmsprop'):
            TensorFlow optimizer name, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/train>`__
            (<span style="color:#C00000"><b>required</b></span> unless given by specification key).
        learning_rate (parameter, float > 0.0): Learning rate
            (<span style="color:#00C000"><b>default</b></span>: 3e-4).
        gradient_norm_clipping (parameter, float > 0.0): Clip gradients by the ratio of the sum
            of their norms (<span style="color:#00C000"><b>default</b></span>: 1.0).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Arguments for the TensorFlow optimizer, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/train>`__.
    """

    tensorflow_optimizers = dict(
        adadelta=tf.train.AdadeltaOptimizer,
        adagrad=tf.train.AdagradOptimizer,
        adam=tf.train.AdamOptimizer,
        gradient_descent=tf.train.GradientDescentOptimizer,
        momentum=tf.train.MomentumOptimizer,
        proximal_adagrad=tf.train.ProximalAdagradOptimizer,
        proximal_gradient_descent=tf.train.ProximalGradientDescentOptimizer,
        rmsprop=tf.train.RMSPropOptimizer
    )
    # tensorflow_optimizers = dict(
    #     adadelta=tf.optimizers.Adadelta,
    #     adagrad=tf.optimizers.Adagrad,
    #     adam=tf.optimizers.Adam,
    #     adamax=tf.optimizers.Adamax,
    #     ftrl=tf.optimizers.Ftrl,
    #     nadam=tf.optimizers.Nadam,
    #     rmsprop=tf.optimizers.RMSprop,
    #     sgd=tf.optimizers.SGD
    # )
    # "clipnorm", "clipvalue", "lr", "decay"}

    def __init__(
        self, name, optimizer, learning_rate=3e-4, gradient_norm_clipping=1.0, summary_labels=None,
        **kwargs
    ):
        super().__init__(name=name, summary_labels=summary_labels)

        assert optimizer in TFOptimizer.tensorflow_optimizers
        self.learning_rate = self.add_module(
            name='learning-rate', module=learning_rate, modules=parameter_modules, dtype='float'
        )
        self.gradient_norm_clipping = self.add_module(
            name='gradient-norm-clipping', module=gradient_norm_clipping,
            modules=parameter_modules, dtype='float'
        )
        self.optimizer = TFOptimizer.tensorflow_optimizers[optimizer]
        self.optimizer_kwargs = kwargs

    def tf_initialize(self):
        super().tf_initialize()

        self.optimizer = self.optimizer(
            learning_rate=self.learning_rate.value, **self.optimizer_kwargs
        )

    def tf_step(self, variables, arguments, fn_loss, fn_initial_gradients=None, **kwargs):
        arguments = util.fmap(function=tf.stop_gradient, xs=arguments)
        loss = fn_loss(**arguments)

        # Force loss value and attached control flow to be computed.
        with tf.control_dependencies(control_inputs=(loss,)):
            # Trivial operation to enforce control dependency
            previous_variables = util.fmap(function=util.identity_operation, xs=variables)

        # Get variables before update.
        with tf.control_dependencies(control_inputs=previous_variables):
            # applied = self.optimizer.minimize(loss=loss, var_list=variables)
            # grads_and_vars = self.optimizer.compute_gradients(loss=loss, var_list=variables)
            # gradients, variables = zip(*grads_and_vars)
            if fn_initial_gradients is None:
                initial_gradients = None
            else:
                initial_gradients = fn_initial_gradients(**arguments)
                initial_gradients = tf.stop_gradient(input=initial_gradients)

            gradients = tf.gradients(ys=loss, xs=variables, grad_ys=initial_gradients)
            assertions = [
                tf.debugging.assert_all_finite(t=gradient, msg='') for gradient in gradients
            ]

        with tf.control_dependencies(control_inputs=assertions):
            gradient_norm_clipping = self.gradient_norm_clipping.value()
            gradients, gradient_norm = tf.clip_by_global_norm(
                t_list=gradients, clip_norm=gradient_norm_clipping
            )
            gradients = self.add_summary(
                label='update-norm', name='gradient-norm-unclipped', tensor=gradient_norm,
                pass_tensors=gradients
            )

            applied = self.optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))

        # Return deltas after actually having change the variables.
        with tf.control_dependencies(control_inputs=(applied,)):
            return [
                variable - previous_variable
                for variable, previous_variable in zip(variables, previous_variables)
            ]

    def get_variables(self, only_trainable=False, only_saved=False):
        variables = super().get_variables(only_trainable=only_trainable, only_saved=only_saved)

        if not only_trainable:
            variables.extend(self.optimizer.variables())

        return variables
