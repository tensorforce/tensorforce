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

from tensorforce import util
from tensorforce.core import parameter_modules, TensorSpec, tf_function, tf_util
from tensorforce.core.objectives import Objective


class PolicyGradient(Objective):
    """
    Policy gradient objective, which maximizes the log-likelihood or likelihood-ratio scaled by the
    target reward value (specification key: `policy_gradient`).

    Args:
        importance_sampling (bool): Whether to use the importance sampling version of the policy
            gradient objective
            (<span style="color:#00C000"><b>default</b></span>: false).
        clipping_value (parameter, float > 0.0): Clipping threshold for the maximized value
            (<span style="color:#00C000"><b>default</b></span>: no clipping).
        early_reduce (bool): Whether to compute objective for aggregated likelihood instead of
            likelihood per action (<span style="color:#00C000"><b>default</b></span>: true).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        reward_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, importance_sampling=False, clipping_value=None, early_reduce=True, name=None,
        states_spec=None, internals_spec=None, auxiliaries_spec=None, actions_spec=None,
        reward_spec=None
    ):
        super().__init__(
            name=name, states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

        self.importance_sampling = importance_sampling

        if clipping_value is None:
            self.clipping_value = None
        else:
            self.clipping_value = self.submodule(
                name='clipping_value', module=clipping_value, modules=parameter_modules, dtype='float',
                min_value=0.0
            )

        self.early_reduce = early_reduce

    def required_policy_fns(self):
        return ('stochastic',)

    def reference_spec(self):
        if self.early_reduce:
            return TensorSpec(type='float', shape=())

        else:
            return TensorSpec(
                type='float', shape=(sum(spec.size for spec in self.actions_spec.values()),)
            )

    @tf_function(num_args=5)
    def reference(self, *, states, horizons, internals, auxiliaries, actions, policy):
        if self.early_reduce:
            log_probability = policy.log_probability(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                actions=actions
            )

        else:
            log_probabilities = policy.log_probabilities(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                actions=actions
            )

            def function(value, spec):
                return tf.reshape(tensor=value, shape=(-1, spec.size))

            log_probabilities = log_probabilities.fmap(
                function=function, zip_values=self.actions_spec
            )
            log_probability = tf.concat(values=tuple(log_probabilities.values()), axis=1)

        return log_probability

    @tf_function(num_args=7)
    def loss(
        self, *, states, horizons, internals, auxiliaries, actions, reward, reference, policy,
        baseline=None
    ):
        log_probability = self.reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, policy=policy
        )

        reference = tf.stop_gradient(input=reference)

        if self.importance_sampling:
            log_ratio = log_probability - reference
            # Clip log_ratio for numerical stability (epsilon < 1.0, hence negative)
            log_epsilon = tf_util.constant(value=np.log(util.epsilon), dtype='float')
            log_ratio = tf.clip_by_value(
                t=log_ratio, clip_value_min=log_epsilon, clip_value_max=-log_epsilon
            )
            target = tf.math.exp(x=log_ratio)
        else:
            target = log_probability

        if not self.early_reduce:
            reward = tf.expand_dims(input=reward, axis=1)

        if self.clipping_value is None:
            scaled_target = target * reward

        else:
            one = tf_util.constant(value=1.0, dtype='float')
            clipping_value = one + self.clipping_value.value()
            if self.importance_sampling:
                min_value = tf.math.reciprocal(x=clipping_value)
                max_value = clipping_value
            else:
                min_value = reference - tf.math.log(x=clipping_value)
                max_value = reference + tf.math.log(x=clipping_value)

            clipped_target = tf.clip_by_value(
                t=target, clip_value_min=min_value, clip_value_max=max_value
            )
            scaled_target = tf.math.minimum(x=(target * reward), y=(clipped_target * reward))

        loss = -scaled_target

        if not self.early_reduce:
            loss = tf.math.reduce_sum(input_tensor=loss, axis=1)

        return loss
