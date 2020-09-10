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

from functools import partial

import tensorflow as tf

from tensorforce.core import parameter_modules, tf_function, tf_util
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
        learning_rate (parameter, float > 0.0): Learning rate
            (<span style="color:#C00000"><b>required</b></span>).
        gradient_norm_clipping (parameter, float > 0.0): Clip gradients by the ratio of the sum
            of their norms (<span style="color:#00C000"><b>default</b></span>: 1.0).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        kwargs: Arguments for the TensorFlow optimizer, special values "decoupled_weight_decay",
            "lookahead" and "moving_average", see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`__
            and `TensorFlow Addons docs
            <https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers>`__.
    """

    def __init__(
        self, *, optimizer, learning_rate, gradient_norm_clipping=None, name=None,
        arguments_spec=None, **kwargs
    ):
        super().__init__(name=name, arguments_spec=arguments_spec)

        assert optimizer in tensorflow_optimizers
        self.tf_optimizer = tensorflow_optimizers[optimizer]

        self.learning_rate = self.submodule(
            name='learning_rate', module=learning_rate, modules=parameter_modules, dtype='float',
            min_value=0.0
        )

        if gradient_norm_clipping is None:
            self.gradient_norm_clipping = None
        else:
            self.gradient_norm_clipping = self.submodule(
                name='gradient_norm_clipping', module=gradient_norm_clipping,
                modules=parameter_modules, dtype='float', min_value=0.0
            )

        self.optimizer_kwargs = kwargs

        def compose(function1, function2):
            def composed(*args, **kwargs):
                return function1(function2(*args, **kwargs))
            return composed

        if 'decoupled_weight_decay' in self.optimizer_kwargs:
            decoupled_weight_decay = self.optimizer_kwargs.pop('decoupled_weight_decay')
            self.tf_optimizer = partial(
                tfa.optimizers.extend_with_decoupled_weight_decay(base_optimizer=self.tf_optimizer),
                weight_decay=decoupled_weight_decay
            )
        if 'lookahead' in self.optimizer_kwargs:
            lookahead = self.optimizer_kwargs.pop('lookahead')
            if isinstance(lookahead, dict) or lookahead is True:
                if lookahead is True:
                    lookahead = dict()
                self.tf_optimizer = compose(
                    function1=partial(tfa.optimizers.Lookahead, name=self.name, **lookahead),
                    function2=self.tf_optimizer
                )
        if 'moving_average' in self.optimizer_kwargs:
            moving_avg = self.optimizer_kwargs.pop('moving_average')
            if isinstance(moving_avg, dict) or moving_avg is True:
                if moving_avg is True:
                    moving_avg = dict()
                self.tf_optimizer = compose(
                    function1=partial(tfa.optimizers.MovingAverage, name=self.name, **moving_avg),
                    function2=self.tf_optimizer
                )

    def initialize(self):
        super().initialize()

        self.tf_optimizer = self.tf_optimizer(
            learning_rate=self.learning_rate.value, name='tf_optimizer', **self.optimizer_kwargs
        )

        self.register_summary(label='update-norm', name='unclipped-gradient-norm')

    def initialize_given_variables(self, *, variables):
        super().initialize_given_variables(variables=variables)

        try:
            self.tf_optimizer._create_all_weights(var_list=variables)
        except AttributeError:
            self.tf_optimizer._create_hypers()
            self.tf_optimizer._create_slots(var_list=variables)

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, fn_loss, **kwargs):
        # Trivial operation to enforce control dependency
        previous_values = list(tf_util.identity(input=variable) for variable in variables)

        # Remember variables before update
        with tf.control_dependencies(control_inputs=previous_values):

            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                for variable in variables:
                    tape.watch(tensor=variable)
                loss = fn_loss(**arguments.to_kwargs())

            gradients = tape.gradient(target=loss, sources=variables)  # , output_gradients=initial

            assertions = list()
            gradients = list(gradients)
            grads_and_vars = list(zip(gradients, variables))
            for n in range(len(gradients) - 1, -1, -1):
                if gradients[n] is None:
                    gradients.pop(n)
                    grads_and_vars.pop(n)
                elif self.config.create_tf_assertions:
                    assertions.append(tf.debugging.assert_all_finite(
                        x=gradients[n], message="Invalid gradient: contains inf or nan."
                    ))
            assert len(gradients) > 0

        with tf.control_dependencies(control_inputs=assertions):

            dependencies = list()
            if self.gradient_norm_clipping is not None:
                clip_norm = self.gradient_norm_clipping.value()
                gradients, grads_norm = tf.clip_by_global_norm(
                    t_list=[tf_util.cast(x=g, dtype='float') for g in gradients],
                    clip_norm=clip_norm
                )
                dependencies.extend(self.summary(
                    label='update-norm', name='unclipped-gradient-norm', data=grads_norm,
                    step='updates'
                ))
                grads_and_vars = [(grad, var) for grad, (_, var) in zip(gradients, grads_and_vars)]

            applied = self.tf_optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            dependencies.append(applied)

        # Return deltas after actually having change the variables.
        with tf.control_dependencies(control_inputs=dependencies):
            return [variable - previous for variable, previous in zip(variables, previous_values)]
