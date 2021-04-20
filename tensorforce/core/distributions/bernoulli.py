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

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules, TensorDict, TensorSpec, TensorsSpec, tf_function, \
    tf_util
from tensorforce.core.distributions import Distribution


class Bernoulli(Distribution):
    """
    Bernoulli distribution, for binary boolean actions (specification key: `bernoulli`).

    Args:
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, name=None, action_spec=None, input_spec=None):
        assert action_spec.type == 'bool'

        parameters_spec = TensorsSpec(
            true_logit=TensorSpec(type='float', shape=action_spec.shape),
            false_logit=TensorSpec(type='float', shape=action_spec.shape),
            probability=TensorSpec(type='float', shape=action_spec.shape),
            state_value=TensorSpec(type='float', shape=action_spec.shape)
        )
        conditions_spec = TensorsSpec()

        super().__init__(
            name=name, action_spec=action_spec, input_spec=input_spec,
            parameters_spec=parameters_spec, conditions_spec=conditions_spec
        )

        if self.input_spec.rank == 1:
            # Single embedding
            action_size = util.product(xs=self.action_spec.shape, empty=0)
            self.logit = self.submodule(
                name='logit', module='linear', modules=layer_modules, size=action_size,
                initialization_scale=0.01, input_spec=self.input_spec
            )

        else:
            # Embedding per action
            if self.input_spec.rank < 1 or self.input_spec.rank > 3:
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
            self.logit = self.submodule(
                name='logit', module='linear', modules=layer_modules, size=size,
                initialization_scale=0.01, input_spec=self.input_spec
            )

    def get_architecture(self):
        return 'Logit:  {}'.format(self.logit.get_architecture())

    def initialize(self):
        super().initialize()

        name = 'distributions/' + self.name + '-probability'
        self.register_summary(label='distribution', name=name)

        spec = self.parameters_spec['probability']
        self.register_tracking(label='distribution', name='probability', spec=spec)

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        one = tf_util.constant(value=1.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        shape = (-1,) + self.action_spec.shape

        # Logit
        logit = self.logit.apply(x=x)
        if self.input_spec.rank == 1:
            logit = tf.reshape(tensor=logit, shape=shape)

        # States value
        state_value = logit

        # Sigmoid for corresponding probability
        probability = tf.sigmoid(x=logit)

        # "Normalized" logits
        true_logit = tf.math.log(x=(probability + epsilon))
        false_logit = tf.math.log(x=(one - probability + epsilon))

        return TensorDict(
            true_logit=true_logit, false_logit=false_logit, probability=probability,
            state_value=state_value
        )

    @tf_function(num_args=1)
    def mode(self, *, parameters, independent):
        probability = parameters['probability']

        # Distribution parameter summaries
        dependencies = list()
        if not independent:
            def fn_summary():
                axis = range(self.action_spec.rank + 1)
                return tf.math.reduce_mean(input_tensor=probability, axis=axis)

            name = 'distributions/' + self.name + '-probability'
            dependencies.extend(self.summary(
                label='distribution', name=name, data=fn_summary, step='timesteps'
            ))

        # Distribution parameter tracking
        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=probability, axis=0)

        dependencies.extend(self.track(label='distribution', name='probability', data=fn_tracking))

        with tf.control_dependencies(control_inputs=dependencies):
            return tf.greater_equal(x=probability, y=tf_util.constant(value=0.5, dtype='float'))

    @tf_function(num_args=2)
    def sample(self, *, parameters, temperature, independent):
        true_logit, false_logit, probability = parameters.get(
            ('true_logit', 'false_logit', 'probability')
        )

        # Distribution parameter summaries
        dependencies = list()
        if not independent:
            def fn_summary():
                axis = range(self.action_spec.rank + 1)
                return tf.math.reduce_mean(input_tensor=probability, axis=axis)

            name = 'distributions/' + self.name + '-probability'
            dependencies.extend(self.summary(
                label='distribution', name=name, data=fn_summary, step='timesteps'
            ))

        # Distribution parameter tracking
        def fn_tracking():
            return tf.math.reduce_mean(input_tensor=probability, axis=0)

        dependencies.extend(self.track(label='distribution', name='probability', data=fn_tracking))

        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        def fn_mode():
            # Deterministic: true if >= 0.5
            half = tf_util.constant(value=0.5, dtype='float')
            return tf.greater_equal(x=probability, y=half)

        def fn_sample():
            # Non-deterministic: sample true if >= uniform distribution
            # Exp numerically stable since logits <= 0.0
            e_true_logit = tf.math.exp(x=(true_logit / (temperature + epsilon)))
            e_false_logit = tf.math.exp(x=(false_logit / (temperature + epsilon)))
            probability = e_true_logit / (e_true_logit + e_false_logit + epsilon)
            uniform = tf.random.uniform(
                shape=tf.shape(input=probability), dtype=tf_util.get_dtype(type='float')
            )
            return tf.greater_equal(x=probability, y=uniform)

        with tf.control_dependencies(control_inputs=dependencies):
            return tf.cond(pred=(temperature < epsilon), true_fn=fn_mode, false_fn=fn_sample)

    @tf_function(num_args=2)
    def log_probability(self, *, parameters, action):
        true_logit, false_logit = parameters.get(('true_logit', 'false_logit'))

        return tf.where(condition=action, x=true_logit, y=false_logit)

    @tf_function(num_args=1)
    def entropy(self, *, parameters):
        true_logit, false_logit, probability = parameters.get(
            ('true_logit', 'false_logit', 'probability')
        )

        one = tf_util.constant(value=1.0, dtype='float')

        return -probability * true_logit - (one - probability) * false_logit

    @tf_function(num_args=2)
    def kl_divergence(self, *, parameters1, parameters2):
        true_logit1, false_logit1, probability1 = parameters1.get(
            ('true_logit', 'false_logit', 'probability')
        )
        true_logit2, false_logit2 = parameters2.get(('true_logit', 'false_logit'))

        true_log_prob_ratio = true_logit1 - true_logit2
        false_log_prob_ratio = false_logit1 - false_logit2

        one = tf_util.constant(value=1.0, dtype='float')

        return probability1 * true_log_prob_ratio + (one - probability1) * false_log_prob_ratio

    @tf_function(num_args=2)
    def action_value(self, *, parameters, action):
        true_logit, false_logit, state_value = parameters.get(
            ('true_logit', 'false_logit', 'state_value')
        )

        logits = tf.where(condition=action, x=true_logit, y=false_logit)

        return state_value + logits

    @tf_function(num_args=1)
    def state_value(self, *, parameters):
        state_value = parameters['state_value']

        return state_value
