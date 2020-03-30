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
from tensorforce.core import tf_function
from tensorforce.core.objectives import Objective


class DeterministicPolicyGradient(Objective):
    """
    Deterministic policy gradient objective (specification key: `det_policy_gradient`).

    Args:
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    @tf_function(num_args=6)
    def loss(self, states, horizons, internals, auxiliaries, actions, reward, policy):
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

    def optimizer_arguments(self, policy, baseline, **kwargs):
        arguments = super().optimizer_arguments()

        def fn_initial_gradients(states, horizons, internals, auxiliaries, actions, reward):
            policy_internals = OrderedDict(
                ((name, internals[name]) for name in policy.internals_spec(policy=policy))
            )
            actions = policy.act(
                states=states, horizons=horizons, internals=policy_internals,
                auxiliaries=auxiliaries, return_internals=False
            )
            baseline_internals = OrderedDict(
                ((name, internals[name]) for name in baseline.internals_spec(policy=baseline))
            )
            actions_value = baseline.actions_value(
                states=states, horizons=horizons, internals=baseline_internals,
                auxiliaries=auxiliaries, actions=actions, reduced=True, return_per_action=False
            )
            assert len(actions) == 1 and len(util.shape(x=list(actions.values())[0])) == 1
            return -tf.gradients(ys=actions_value, xs=list(actions.values()))[0][0]

        arguments['fn_initial_gradients'] = fn_initial_gradients

        return arguments
