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
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, summary_labels=None, name=None, action_spec=None, input_spec=None):
        assert action_spec.type == 'bool'

        parameters_spec = TensorsSpec(
            true_logit=TensorSpec(type='float', shape=action_spec.shape),
            false_logit=TensorSpec(type='float', shape=action_spec.shape),
            probability=TensorSpec(type='float', shape=action_spec.shape),
            states_value=TensorSpec(type='float', shape=action_spec.shape)
        )
        conditions_spec = TensorsSpec()

        super().__init__(
            summary_labels=summary_labels, name=name, action_spec=action_spec,
            input_spec=input_spec, parameters_spec=parameters_spec, conditions_spec=conditions_spec
        )

        if len(self.input_spec.shape) == 1:
            # Single embedding
            action_size = util.product(xs=self.action_spec.shape, empty=0)
            self.logit = self.add_module(
                name='logit', module='linear', modules=layer_modules, size=action_size,
                input_spec=self.input_spec
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
            self.logit = self.add_module(
                name='logit', module='linear', modules=layer_modules, size=size,
                input_spec=self.input_spec
            )

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        one = tf_util.constant(value=1.0, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')
        shape = (-1,) + self.action_spec.shape

        # Logit
        logit = self.logit.apply(x=x)
        if len(self.input_spec.shape) == 1:
            logit = tf.reshape(tensor=logit, shape=shape)

        # States value
        states_value = logit

        # Sigmoid for corresponding probability
        probability = tf.sigmoid(x=logit)

        # Clip probability for numerical stability
        probability = tf.clip_by_value(
            t=probability, clip_value_min=epsilon, clip_value_max=(one - epsilon)
        )

        # "Normalized" logits
        true_logit = tf.math.log(x=probability)
        false_logit = tf.math.log(x=(one - probability))

        return TensorDict(
            true_logit=true_logit, false_logit=false_logit, probability=probability,
            states_value=states_value
        )

    @tf_function(num_args=2)
    def sample(self, *, parameters, temperature):
        true_logit, false_logit, probability = parameters.get(
            ('true_logit', 'false_logit', 'probability')
        )

        summary_probability = probability
        for _ in range(len(self.action_spec.shape)):
            summary_probability = tf.math.reduce_mean(input_tensor=summary_probability, axis=1)

        true_logit, false_logit, probability = self.add_summary(
            label=('distributions', 'bernoulli'), name='probability', tensor=summary_probability,
            pass_tensors=(true_logit, false_logit, probability)
        )

        half = tf_util.constant(value=0.5, dtype='float')
        epsilon = tf_util.constant(value=util.epsilon, dtype='float')

        # Deterministic: true if >= 0.5
        definite = tf.greater_equal(x=probability, y=half)

        # Non-deterministic: sample true if >= uniform distribution
        e_true_logit = tf.math.exp(x=(true_logit / temperature))
        e_false_logit = tf.math.exp(x=(false_logit / temperature))
        probability = e_true_logit / (e_true_logit + e_false_logit)
        uniform = tf.random.uniform(
            shape=tf.shape(input=probability), dtype=tf_util.get_dtype(type='float')
        )
        sampled = tf.greater_equal(x=probability, y=uniform)

        return tf.where(condition=(temperature < epsilon), x=definite, y=sampled)

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

    @tf_function(num_args=1)
    def states_value(self, *, parameters):
        states_value = parameters['states_value']

        return states_value

    @tf_function(num_args=2)
    def action_value(self, *, parameters, action):
        true_logit, false_logit, states_value = parameters.get(
            ('true_logit', 'false_logit', 'states_value')
        )

        logits = tf.where(condition=action, x=true_logit, y=false_logit)

        return states_value + logits

    @tf_function(num_args=1)
    def all_action_values(self, *, parameters):
        true_logit, false_logit, states_value = parameters.get(
            ('true_logit', 'false_logit', 'states_value')
        )

        logits = tf.stack(values=(false_logit, true_logit), axis=-1)
        states_value = tf.expand_dims(input=states_value, axis=-1)

        return states_value + logits
