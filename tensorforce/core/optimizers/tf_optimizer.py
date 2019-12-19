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
from tensorforce.core import parameter_modules
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
except BaseException:
    pass


class TFOptimizer(Optimizer):
    """
    TensorFlow optimizer (specification key: `tf_optimizer`, `adadelta`, `adagrad`, `adam`,
    `adamax`, `adamw`, `ftrl`, `lazyadam`, `nadam`, `radam`, `ranger`, `rmsprop`, `sgd`, `sgdw`)

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer (`adadelta` | `adagrad` | `adam` | `adamax` | `adamw` | `ftrl` | `lazyadam` | `nadam` | `radam` | `ranger` | `rmsprop` | `sgd` | `sgdw`):
            TensorFlow optimizer name, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`__
            and `TensorFlow Addons docs
            <https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers>`__
            (<span style="color:#C00000"><b>required</b></span> unless given by specification key).
        learning_rate (parameter, float > 0.0): Learning rate
            (<span style="color:#00C000"><b>default</b></span>: 3e-4).
        gradient_norm_clipping (parameter, float > 0.0): Clip gradients by the ratio of the sum
            of their norms (<span style="color:#00C000"><b>default</b></span>: 1.0).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Arguments for the TensorFlow optimizer, special values "decoupled_weight_decay",
            "lookahead" and "moving_average", see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`__
            and `TensorFlow Addons docs
            <https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers>`__.
    """

    def __init__(
        self, name, optimizer, learning_rate=3e-4, gradient_norm_clipping=1.0, summary_labels=None,
        **kwargs
    ):
        super().__init__(name=name, summary_labels=summary_labels)

        assert optimizer in tensorflow_optimizers
        self.optimizer = tensorflow_optimizers[optimizer]
        self.learning_rate = self.add_module(
            name='learning-rate', module=learning_rate, modules=parameter_modules, dtype='float'
        )
        self.gradient_norm_clipping = self.add_module(
            name='gradient-norm-clipping', module=gradient_norm_clipping,
            modules=parameter_modules, dtype='float'
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
                tf.debugging.assert_all_finite(x=gradient, message='') for gradient in gradients
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
        optimizer = self.optimizer
        while True:
            for variable in optimizer.weights:
                name = '/' + self.name + '/'
                if name in variable.name:
                    name = variable.name[variable.name.rindex(name) + len(name): -2]
                else:
                    name = variable.name[variable.name.rindex('/') + 1: -2]
                self.variables[name] = variable
            for name, value in optimizer._hyper.items():
                if isinstance(value, tf.Variable):
                    self.variables[name] = value
            if hasattr(optimizer, '_ema'):
                for variable in optimizer._ema._averages.values():
                    assert variable.name.startswith('agent/') and \
                        variable.name.endswith('/ExponentialMovingAverage:0')
                    self.variables[variable.name[:-2]] = variable
            if hasattr(optimizer, '_optimizer'):
                optimizer = optimizer._optimizer
            else:
                break

        variables = super().get_variables(only_trainable=only_trainable, only_saved=only_saved)

        return variables
