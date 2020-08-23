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

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules,  TensorDict, TensorSpec, TensorsSpec, tf_function, \
    tf_util
from tensorforce.core.distributions import Distribution


class Beta(Distribution):
    """
    Beta distribution, for bounded continuous actions (specification key: `beta`).

    Args:
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, name=None, action_spec=None, input_spec=None):
        assert action_spec.type == 'float' and action_spec.min_value is not None and \
            action_spec.max_value is not None

        parameters_spec = TensorsSpec(
            alpha=TensorSpec(type='float', shape=action_spec.shape),
            beta=TensorSpec(type='float', shape=action_spec.shape),
            alpha_beta=TensorSpec(type='float', shape=action_spec.shape),
            log_norm=TensorSpec(type='float', shape=action_spec.shape)
        )
        conditions_spec = TensorsSpec()

        super().__init__(
            name=name, action_spec=action_spec, input_spec=input_spec,
            parameters_spec=parameters_spec, conditions_spec=conditions_spec
        )

        if len(self.input_spec.shape) == 1:
            # Single embedding
            action_size = util.product(xs=self.action_spec.shape, empty=0)
            self.alpha = self.submodule(
                name='alpha', module='linear', modules=layer_modules, size=action_size,
                initialization_scale=0.01, input_spec=self.input_spec
            )
            self.beta = self.submodule(
                name='beta', module='linear', modules=layer_modules, size=action_size,
                initialization_scale=0.01, input_spec=self.input_spec
            )

        else:
            # Embedding per action
            if len(self.input_spec.shape) < 1 or len(self.input_spec.shape) > 3:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.input_spec.shape,
                    hint='invalid rank'
                )
            if self.input_spec.shape[:-1] == self.action_spec.shape[:-1]:
                size = self.action_spec.shape[-1]
            elif self.input_spec.shape[:-1] == self.action_spec.shape:
                size = 0
            else:
                raise TensorforceError.value(
                    name=name, argument='input_spec.shape', value=self.input_spec.shape,
                    hint='not flattened and incompatible with action shape'
                )
            self.alpha = self.submodule(
                name='alpha', module='linear', modules=layer_modules, size=size,
                initialization_scale=0.01, input_spec=self.input_spec
            )
            self.beta = self.submodule(
                name='beta', module='linear', modules=layer_modules, size=size,
                initialization_scale=0.01, input_spec=self.input_spec
            )

    def initialize(self):
        super().initialize()

        prefix = 'distributions/' + self.name
        self.register_summary(label='distribution', name=(prefix + '-alpha', prefix + '-beta'))

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        # Softplus to ensure alpha and beta >= 1
        one = tf_util.constant(value=1.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        log_epsilon = tf_util.constant(value=np.log(util.epsilon), dtype='float')
        shape = (-1,) + self.action_spec.shape

        # Alpha
        alpha = self.alpha.apply(x=x)
        # epsilon < 1.0, hence negative
        alpha = tf.clip_by_value(t=alpha, clip_value_min=log_epsilon, clip_value_max=-log_epsilon)
        alpha = tf.math.exp(x=alpha) + one  # tf.math.softplus(features=beta) ???
        if len(self.input_spec.shape) == 1:
            alpha = tf.reshape(tensor=alpha, shape=shape)

        # Beta
        beta = self.beta.apply(x=x)
        # epsilon < 1.0, hence negative
        beta = tf.clip_by_value(t=beta, clip_value_min=log_epsilon, clip_value_max=-log_epsilon)
        beta = tf.math.exp(x=beta) + one  # tf.math.softplus(features=beta) ???
        if len(self.input_spec.shape) == 1:
            beta = tf.reshape(tensor=beta, shape=shape)

        # Alpha + Beta
        alpha_beta = tf.maximum(x=(alpha + beta), y=epsilon)

        # Log norm
        log_norm = tf.math.lgamma(x=alpha) + tf.math.lgamma(x=beta) - tf.math.lgamma(x=alpha_beta)

        return TensorDict(alpha=alpha, beta=beta, alpha_beta=alpha_beta, log_norm=log_norm)

    @tf_function(num_args=1)
    def mode(self, *, parameters):
        beta, alpha_beta = parameters.get(('beta', 'alpha_beta'))

        action = beta / alpha_beta

        min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
        max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')

        return min_value + (max_value - min_value) * action

    @tf_function(num_args=2)
    def sample(self, *, parameters, temperature):
        alpha, beta, alpha_beta, log_norm = parameters.get(
            ('alpha', 'beta', 'alpha_beta', 'log_norm')
        )

        # Distribution parameter summaries
        def fn_summary():
            return tf.math.reduce_mean(input_tensor=alpha, axis=range(self.action_spec.rank + 1)), \
                tf.math.reduce_mean(input_tensor=beta, axis=range(self.action_spec.rank + 1))

        prefix = 'distributions/' + self.name
        dependencies = self.summary(
            label='distribution', name=(prefix + '-alpha', prefix + '-beta'), data=fn_summary,
            step='timesteps'
        )

        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        def fn_mode():
            # Deterministic: mean as action
            return beta / alpha_beta

        def fn_sample():
            # Non-deterministic: sample action using gamma distribution
            alpha_sample = tf.random.gamma(shape=(), alpha=alpha, dtype=tf_util.get_dtype(type='float'))
            beta_sample = tf.random.gamma(shape=(), alpha=beta, dtype=tf_util.get_dtype(type='float'))
            return beta_sample / tf.maximum(x=(alpha_sample + beta_sample), y=epsilon)

        action = tf.cond(pred=(temperature < epsilon), true_fn=fn_mode, false_fn=fn_sample)

        min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
        max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')

        with tf.control_dependencies(control_inputs=dependencies):
            return min_value + (max_value - min_value) * action

    @tf_function(num_args=2)
    def log_probability(self, *, parameters, action):
        alpha, beta, log_norm = parameters.get(('alpha', 'beta', 'log_norm'))

        min_value = tf_util.constant(value=self.action_spec.min_value, dtype='float')
        max_value = tf_util.constant(value=self.action_spec.max_value, dtype='float')

        action = (action - min_value) / (max_value - min_value)

        one = tf_util.constant(value=1.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        action = tf.minimum(x=action, y=(one - epsilon))

        return tf.math.xlogy(x=(beta - one), y=tf.maximum(x=action, y=epsilon)) + \
            (alpha - one) * tf.math.log1p(x=(-action)) - log_norm

    @tf_function(num_args=1)
    def entropy(self, *, parameters):
        alpha, beta, alpha_beta, log_norm = parameters.get(
            ('alpha', 'beta', 'alpha_beta', 'log_norm')
        )

        one = tf_util.constant(value=1.0, dtype='float')

        digamma_alpha = tf_util.cast(x=tf.math.digamma(x=tf_util.float32(x=alpha)), dtype='float')
        digamma_beta = tf_util.cast(x=tf.math.digamma(x=tf_util.float32(x=beta)), dtype='float')
        digamma_alpha_beta = tf_util.cast(
            x=tf.math.digamma(x=tf_util.float32(x=alpha_beta)), dtype='float'
        )

        return log_norm - (beta - one) * digamma_beta - (alpha - one) * digamma_alpha + \
            (alpha_beta - one - one) * digamma_alpha_beta

    @tf_function(num_args=2)
    def kl_divergence(self, *, parameters1, parameters2):
        alpha1, beta1, alpha_beta1, log_norm1 = parameters1.get(
            ('alpha', 'beta', 'alpha_beta', 'log_norm')
        )
        alpha2, beta2, alpha_beta2, log_norm2 = parameters2.get(
            ('alpha', 'beta', 'alpha_beta', 'log_norm')
        )

        digamma_alpha1 = tf_util.cast(x=tf.math.digamma(x=tf_util.float32(x=alpha1)), dtype='float')
        digamma_beta1 = tf_util.cast(x=tf.math.digamma(x=tf_util.float32(x=beta1)), dtype='float')
        digamma_alpha_beta1 = tf_util.cast(
            x=tf.math.digamma(x=tf_util.float32(x=alpha_beta1)), dtype='float'
        )

        return log_norm2 - log_norm1 - digamma_beta1 * (beta2 - beta1) - \
            digamma_alpha1 * (alpha2 - alpha1) + digamma_alpha_beta1 * \
            (alpha_beta2 - alpha_beta1)
