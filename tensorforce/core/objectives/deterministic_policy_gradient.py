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

from tensorforce import util
from tensorforce.core import TensorDict, tf_function, tf_util
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
        reward_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    @tf_function(num_args=6)
    def loss(self, *, states, horizons, internals, auxiliaries, actions, reward, policy):
        policy_actions = policy.act(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            deterministic=True, return_internals=False
        )

        summed_actions = list()
        for name, spec, action in util.zip_items(policy.actions_spec, policy_actions):
            rank = len(spec.shape)
            for n in range(rank):
                action = tf.math.reduce_sum(input_tensor=action, axis=(rank - n))
            summed_actions.append(action)
        summed_actions = tf.math.add_n(inputs=summed_actions)
        # mean? (will be mean later)
        # tf.concat(values=, axis=1)
        # tf.math.reduce_mean(input_tensor=, axis=1)

        return summed_actions

    def optimizer_arguments(self, *, policy, baseline, **kwargs):
        arguments = super().optimizer_arguments()

        def fn_initial_gradients(*, states, horizons, internals, auxiliaries, actions, reward):
            if 'policy' in internals:
                policy_internals = internals['policy']
                baseline_internals = internals['baseline']
            else:
                policy_internals = internals
                # TODO: Baseline currently cannot have internal states, since generally only policy
                # internals are passed to policy optimizer
                assert len(baseline.internals_spec) == 0
                baseline_internals = TensorDict()

            actions = policy.act(
                states=states, horizons=horizons, internals=policy_internals,
                auxiliaries=auxiliaries,
                 deterministic=True, return_internals=False
            )
            action = actions.value()

            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(tensor=action)
                assert len(actions) == 1 and len(tf_util.shape(x=action)) == 1
                actions_value = baseline.actions_value(
                    states=states, horizons=horizons, internals=baseline_internals,
                    auxiliaries=auxiliaries, actions=actions, reduced=True, return_per_action=False
                )

            return -tape.gradient(target=actions_value, sources=action)[0]

        arguments['fn_initial_gradients'] = fn_initial_gradients

        return arguments
