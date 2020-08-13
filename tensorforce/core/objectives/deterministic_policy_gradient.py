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

from tensorforce.core import TensorDict, TensorSpec, tf_function, tf_util
from tensorforce.core.objectives import Objective


class DeterministicPolicyGradient(Objective):
    """
    Deterministic policy gradient objective (specification key: `det_policy_gradient`).

    Args:
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        reward_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, name=None, states_spec=None, internals_spec=None, auxiliaries_spec=None,
        actions_spec=None, reward_spec=None
    ):
        super().__init__(
            name=name, states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

        # TODO: is singleton and single value required?
        if not self.actions_spec.is_singleton():
            raise TensorforceError.value(
                name='DeterministicPolicyGradient', argument='actions', value=actions,
                hint='is not a singleton action specification'
            )
        action_spec = self.actions_spec.singleton()
        if action_spec.type != 'float':
            raise TensorforceError.value(
                name='DeterministicPolicyGradient', argument='actions', value=actions,
                hint='is not a float action'
            )
        elif (action_spec.shape != () and action_spec.shape != (1,)):
            raise TensorforceError.value(
                name='DeterministicPolicyGradient', argument='actions', value=actions,
                hint='consists of more than a single action value'
            )

    def required_policy_fns(self):
        return ('policy',)

    def required_baseline_fns(self):
        return ('action_value',)

    def reference_spec(self):
        return TensorSpec(type='float', shape=())

    def optimizer_arguments(self, *, policy, baseline, **kwargs):
        arguments = super().optimizer_arguments()

        def fn_initial_gradients(
            *, states, horizons, internals, auxiliaries, actions, reward, reference
        ):
            policy_internals = internals['policy']
            baseline_internals = internals['baseline']
            if not self.parent.separate_baseline:
                # TODO: Baseline currently cannot have internal states, since generally only policy
                # internals are passed to policy optimizer
                assert len(baseline.internals_spec) == 0

            actions, _ = policy.act(
                states=states, horizons=horizons, internals=policy_internals,
                auxiliaries=auxiliaries, independent=True
            )

            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                action = actions.singleton()
                tape.watch(tensor=action)

                action_value = baseline.action_value(
                    states=states, horizons=horizons, internals=baseline_internals,
                    auxiliaries=auxiliaries, actions=actions
                )

                # TODO: is singleton and single value required?
                action_spec = self.actions_spec.singleton()
                if len(action_spec.shape) == 0:
                    return -tape.gradient(target=action_value, sources=action)[0]
                elif len(action_spec.singleton().shape) == 1:
                    return -tape.gradient(target=action_value, sources=action)[0][0]

        arguments['fn_initial_gradients'] = fn_initial_gradients

        return arguments


    @tf_function(num_args=5)
    def reference(self, *, states, horizons, internals, auxiliaries, actions, policy):
        policy_action, _ = policy.act(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            independent=True
        )

        policy_action = policy_action.singleton()
        if len(self.actions_spec.singleton().shape) == 1:
            policy_action = tf.squeeze(input=policy_action, axis=1)

        return policy_action

    @tf_function(num_args=7)
    def loss(self, *, states, horizons, internals, auxiliaries, actions, reward, policy, reference):
        policy_action = self.reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, policy=policy
        )

        # reference = tf.stop_gradient(input=reference)

        # summed_actions = list()
        # for name, spec, action in self.actions_spec.zip_items(policy_action):
        #     for n in range(spec.rank):
        #         action = tf.math.reduce_sum(input_tensor=action, axis=(spec.rank - n))
        #     summed_actions.append(action)
        # summed_actions = tf.math.add_n(inputs=summed_actions)

        return policy_action
