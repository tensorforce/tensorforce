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
from tensorforce.core import tf_function
from tensorforce.core.objectives import Objective


class DeterministicPolicyGradient(Objective):
    """
    Deterministic policy gradient objective (specification key: `det_policy_gradient`).

    Args:
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def input_signature(self, function):
        if function == 'initial_gradients':
            return [
                util.to_tensor_spec(value_spec=self.parent.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.parent.internals_spec, batched=True),
                util.to_tensor_spec(value_spec=self.parent.auxiliaries_spec, batched=True)
            ]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=6)
    def tf_loss_per_instance(self, states, horizons, internals, auxiliaries, actions, reward):
        policy_actions = policy.act(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            return_internals=False
        )

        summed_actions = list()
        for name, action in policy_actions.items():
            rank = len(policy.actions_spec[name]['shape'])
            for n in range(rank):
                action = tf.math.reduce_sum(input_tensor=action, axis=(rank - n))
            summed_actions.append(action)
        summed_actions = tf.math.add_n(inputs=summed_actions)
        # mean? (will be mean later)
        # tf.concat(values=, axis=1)
        # tf.math.reduce_mean(input_tensor=, axis=1)

        return summed_actions

    @tf_function(num_args=4)
    def tf_initial_gradients(self, states, horizons, internals, auxiliaries):
        actions = policy.act(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            return_internals=False
        )
        actions_value = baseline.actions_value(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reduced=True, return_per_action=False
        )
        assert len(actions) == 1 and len(util.shape(x=list(actions.values())[0])) == 1
        gradients = -tf.gradients(ys=actions_value, xs=list(actions.values()))[0][0]

        return gradients

    def optimizer_arguments(self, policy, baseline, **kwargs):
        arguments = super().optimizer_arguments()

        def fn_initial_gradients(states, horizons, internals, auxiliaries, actions, reward):
            return self.initial_gradients(
                policy=policy, baseline=baseline, states=states, horizons=horizons,
                internals=internals, auxiliaries=auxiliaries
            )

        arguments['fn_initial_gradients'] = fn_initial_gradients

        return arguments
