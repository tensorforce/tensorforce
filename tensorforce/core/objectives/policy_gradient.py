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

from collections import OrderedDict

import tensorflow as tf

from tensorforce import util
from tensorforce.core import parameter_modules, tf_function
from tensorforce.core.objectives import Objective


class PolicyGradient(Objective):
    """
    Policy gradient objective, which maximizes the log-likelihood or likelihood-ratio scaled by the
    target reward value (specification key: `policy_gradient`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        ratio_based (bool): Whether to scale the likelihood-ratio instead of the log-likelihood
            (<span style="color:#00C000"><b>default</b></span>: false).
        clipping_value (parameter, float >= 0.0): Clipping threshold for the maximized value
            (<span style="color:#00C000"><b>default</b></span>: no clipping).
        early_reduce (bool): Whether to compute objective for reduced likelihoods instead of per
            likelihood (<span style="color:#00C000"><b>default</b></span>: true).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, ratio_based=False, clipping_value=0.0, early_reduce=False, summary_labels=None,
        name=None, states_spec=None, internals_spec=None, auxiliaries_spec=None, actions_spec=None
    ):
        super().__init__(
            summary_labels=summary_labels, name=name, states_spec=states_spec,
            internals_spec=internals_spec, auxiliaries_spec=auxiliaries_spec,
            actions_spec=actions_spec
        )

        self.ratio_based = ratio_based

        clipping_value = 0.0 if clipping_value is None else clipping_value
        self.clipping_value = self.add_module(
            name='clipping_value', module=clipping_value, modules=parameter_modules, dtype='float',
            min_value=0.0
        )

        self.early_reduce = early_reduce

    def reference_spec(self):
        if not self.ratio_based:
            return super().reference_spec()

        elif self.early_reduce:
            return dict(type='float', shape=())

        else:
            num_actions = 0
            for spec in self.parent.actions_spec.values():
                num_actions += util.product(xs=spec['shape'])
            return dict(type='float', shape=(num_actions,))

    @tf_function(num_args=6)
    def reference(self, states, horizons, internals, auxiliaries, actions, reward, policy):
        if self.ratio_based:
            internals = OrderedDict(
                ((name, internals[name]) for name in policy.internals_spec(policy=policy))
            )
            return policy.log_probability(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                actions=actions, reduced=self.early_reduce, return_per_action=False
            )

        else:
            return super().reference(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                actions=actions, reward=reward, policy=policy
            )

    @tf_function(num_args=7)
    def comparative_loss(
        self, states, horizons, internals, auxiliaries, actions, reward, reference, policy
    ):
        internals = OrderedDict(
            ((name, internals[name]) for name in policy.internals_spec(policy=policy))
        )
        log_probability = policy.log_probability(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reduced=self.early_reduce, return_per_action=False
        )

        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        one = tf.constant(value=1.0, dtype=util.tf_dtype(dtype='float'))
        clipping_value = self.clipping_value.value()

        if self.ratio_based:
            scaling = tf.exp(x=(log_probability - tf.stop_gradient(input=reference)))
            min_value = one / (one + clipping_value)
            max_value = one + clipping_value

        else:
            scaling = log_probability
            min_value = -clipping_value
            max_value = log_probability + one

        if not self.early_reduce:
            reward = tf.expand_dims(input=reward, axis=1)

        def no_clipping():
            return scaling * reward

        def apply_clipping():
            clipped_scaling = tf.clip_by_value(
                t=scaling, clip_value_min=min_value, clip_value_max=max_value
            )
            return tf.minimum(x=(scaling * reward), y=(clipped_scaling * reward))

        skip_clipping = tf.math.equal(x=clipping_value, y=zero)
        scaled = self.cond(pred=skip_clipping, true_fn=no_clipping, false_fn=apply_clipping)

        loss = -scaled

        if not self.early_reduce:
            loss = tf.math.reduce_mean(input_tensor=loss, axis=1)

        return loss
