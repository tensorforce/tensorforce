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

from functools import partial

import tensorflow as tf

from tensorforce import util
from tensorforce.core import parameter_modules, tf_function
from tensorforce.core.optimizers import Optimizer


tensorflow_optimizers = dict(
    adadelta=tf.keras.optimizers.Adadelta,
    adagrad=tf.keras.optimizers.Adagrad,
    adam=tf.keras.optimizers.Adam,
    adamax=tf.keras.optimizers.Adamax,
    ftrl=tf.keras.optimizers.Ftrl,
    nadam=tf.keras.optimizers.Nadam,
    rmsprop=tf.keras.optimizers.RMSprop,
    sgd=tf.keras.optimizers.SGD
)


try:
    import tensorflow_addons as tfa

    tensorflow_optimizers['adamw'] = tfa.optimizers.AdamW
    tensorflow_optimizers['lazyadam'] = tfa.optimizers.LazyAdam
    tensorflow_optimizers['radam'] = tfa.optimizers.RectifiedAdam
    tensorflow_optimizers['ranger'] = (lambda **kwargs: tfa.optimizers.Lookahead(
        optimizer=tfa.optimizers.RectifiedAdam(**kwargs), name=kwargs['name']
    ))
    tensorflow_optimizers['sgdw'] = tfa.optimizers.SGDW
except ModuleNotFoundError:
    pass


class TFOptimizer(Optimizer):
    """
    TensorFlow optimizer (specification key: `tf_optimizer`, `adadelta`, `adagrad`, `adam`,
    `adamax`, `adamw`, `ftrl`, `lazyadam`, `nadam`, `radam`, `ranger`, `rmsprop`, `sgd`, `sgdw`)

    Args:
        optimizer (`adadelta` | `adagrad` | `adam` | `adamax` | `adamw` | `ftrl` | `lazyadam` | `nadam` | `radam` | `ranger` | `rmsprop` | `sgd` | `sgdw`):
            TensorFlow optimizer name, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`__
            and `TensorFlow Addons docs
            <https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers>`__
            (<span style="color:#C00000"><b>required</b></span> unless given by specification key).
        learning_rate (parameter, float >= 0.0): Learning rate
            (<span style="color:#00C000"><b>default</b></span>: 3e-4).
        gradient_norm_clipping (parameter, float >= 0.0): Clip gradients by the ratio of the sum
            of their norms (<span style="color:#00C000"><b>default</b></span>: 1.0).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        optimized_module (module): <span style="color:#0000C0"><b>internal use</b></span>.
        kwargs: Arguments for the TensorFlow optimizer, special values "decoupled_weight_decay",
            "lookahead" and "moving_average", see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`__
            and `TensorFlow Addons docs
            <https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers>`__.
    """

    def __init__(
        self, optimizer, learning_rate=3e-4, gradient_norm_clipping=1.0, summary_labels=None,
        name=None, states_spec=None, internals_spec=None, auxiliaries_spec=None, actions_spec=None,
        optimized_module=None, **kwargs
    ):
        super().__init__(
            summary_labels=summary_labels, name=name, states_spec=states_spec,
            internals_spec=internals_spec, auxiliaries_spec=auxiliaries_spec,
            actions_spec=actions_spec, optimized_module=optimized_module
        )

        assert optimizer in tensorflow_optimizers
        self.optimizer = tensorflow_optimizers[optimizer]
        self.learning_rate = self.add_module(
            name='learning_rate', module=learning_rate, modules=parameter_modules, dtype='float',
            min_value=0.0
        )
        self.gradient_norm_clipping = self.add_module(
            name='gradient_norm_clipping', module=gradient_norm_clipping,
            modules=parameter_modules, dtype='float', min_value=0.0
        )
        self.optimizer_kwargs = kwargs

        if 'decoupled_weight_decay' in self.optimizer_kwargs:
            decoupled_weight_decay = self.optimizer_kwargs.pop('decoupled_weight_decay')
            self.optimizer = partial(
                tfa.optimizers.extend_with_decoupled_weight_decay(base_optimizer=self.optimizer),
                weight_decay=decoupled_weight_decay
            )
        if 'lookahead' in self.optimizer_kwargs:
            lookahead = self.optimizer_kwargs.pop('lookahead')
            if isinstance(lookahead, dict) or lookahead is True:
                if lookahead is True:
                    lookahead = dict()
                self.optimizer = util.compose(
                    function1=partial(tfa.optimizers.Lookahead, name=self.name, **lookahead),
                    function2=self.optimizer
                )
        if 'moving_average' in self.optimizer_kwargs:
            moving_avg = self.optimizer_kwargs.pop('moving_average')
            if isinstance(moving_avg, dict) or moving_avg is True:
                if moving_avg is True:
                    moving_avg = dict()
                self.optimizer = util.compose(
                    function1=partial(tfa.optimizers.MovingAverage, name=self.name, **moving_avg),
                    function2=self.optimizer
                )

    def tf_initialize(self):
        super().tf_initialize()

        self.optimizer = self.optimizer(
            learning_rate=self.learning_rate.value, name=self.name, **self.optimizer_kwargs
        )

    @tf_function(num_args=1)
    def step(self, arguments, variables, fn_loss, fn_initial_gradients=None, **kwargs):
        arguments = util.fmap(function=tf.stop_gradient, xs=arguments)

        loss = fn_loss(**arguments)

        # Force loss value and attached control flow to be computed.
        with tf.control_dependencies(control_inputs=(loss,)):
            # Trivial operation to enforce control dependency
            previous_variables = util.fmap(function=util.identity_operation, xs=variables)

        # Get variables before update.
        with tf.control_dependencies(control_inputs=previous_variables):
            if fn_initial_gradients is None:
                initial_gradients = None
            else:
                initial_gradients = fn_initial_gradients(**arguments)
                initial_gradients = tf.stop_gradient(input=initial_gradients)

            gradients = tf.gradients(ys=loss, xs=variables, grad_ys=initial_gradients)

            actual_variables = list()
            actual_gradients = list()
            for variable, gradient in zip(variables, gradients):
                if gradient is not None:
                    actual_variables.append(variable)
                    actual_gradients.append(gradient)
            assert len(actual_variables) > 0

            assertions = [
                tf.debugging.assert_all_finite(x=gradient, message="Finite gradients check.")
                for gradient in actual_gradients if gradient is not None
            ]

        with tf.control_dependencies(control_inputs=assertions):
            gradient_norm_clipping = self.gradient_norm_clipping.value()
            gradients, gradient_norm = tf.clip_by_global_norm(
                t_list=gradients, clip_norm=gradient_norm_clipping
            )
            actual_gradients = self.add_summary(
                label='update-norm', name='gradient-norm-unclipped', tensor=gradient_norm,
                pass_tensors=actual_gradients
            )

            applied = self.optimizer.apply_gradients(
                grads_and_vars=zip(actual_gradients, actual_variables)
            )

        # Return deltas after actually having change the variables.
        with tf.control_dependencies(control_inputs=(applied,)):
            return [
                variable - previous_variable
                for variable, previous_variable in zip(variables, previous_variables)
            ]
