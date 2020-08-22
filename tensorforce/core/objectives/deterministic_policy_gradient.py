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

from tensorforce import TensorforceError
from tensorforce.core import TensorSpec, tf_function, tf_util
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

        if not all(spec.type == 'float' for spec in self.actions_spec.values()):
            raise TensorforceError.value(
                name='DeterministicPolicyGradient', argument='actions', value=self.actions_spec,
                hint='is not a float action'
            )

    def required_policy_fns(self):
        return ('policy',)

    def required_baseline_fns(self):
        return ('action_value',)

    def reference_spec(self):
        # return self.actions_spec
        return TensorSpec(type='float', shape=())

    @tf_function(num_args=5)
    def reference(self, *, states, horizons, internals, auxiliaries, actions, policy):
        # deterministic = tf_util.constant(value=True, dtype='bool')
        # return policy.act(
        #     states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
        #     deterministic=deterministic, independent=True
        # )

        return tf_util.zeros(shape=(tf.shape(input=actions.value())[0],), dtype='float')

    def loss(
        self, *, states, horizons, internals, auxiliaries, actions, reward, reference, policy,
        baseline
    ):
        # actions = self.reference(
        #     states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
        #     actions=actions, policy=policy
        # )

        deterministic = tf_util.constant(value=True, dtype='bool')
        actions, _ = policy.act(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            deterministic=deterministic, independent=True
        )

        action_value = baseline.action_value(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions
        )

        return -action_value
