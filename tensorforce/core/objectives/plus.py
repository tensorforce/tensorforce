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
import tensorforce.core
from tensorforce.core import TensorSpec, tf_function, tf_util
from tensorforce.core.objectives import Objective


class Plus(Objective):
    """
    Additive combination of two objectives (specification key: `plus`).

    Args:
        objective1 (specification): First objective configuration
            (<span style="color:#C00000"><b>required</b></span>).
        objective2 (specification): Second objective configuration
            (<span style="color:#C00000"><b>required</b></span>).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        reward_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, objective1, objective2, name=None, states_spec=None,
        internals_spec=None, auxiliaries_spec=None, actions_spec=None, reward_spec=None
    ):
        super().__init__(
            name=name, states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

        self.objective1 = self.submodule(
            name='objective1', module=objective1, modules=tensorforce.core.objective_modules,
            states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

        self.objective2 = self.submodule(
            name='objective2', module=objective2, modules=tensorforce.core.objective_modules,
            states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if hasattr(self, 'objective1') and name in (
            'states_spec', 'internals_spec', 'auxiliaries_spec', 'actions_spec', 'reward_spec'
        ):
            self.objective1.__setattr__(name, value)
            self.objective2.__setattr__(name, value)

    def required_policy_fns(self):
        return self.objective1.required_policy_fns() + self.objective2.required_policy_fns()

    def required_baseline_fns(self):
        return self.objective1.required_baseline_fns() + self.objective2.required_baseline_fns()

    def reference_spec(self):
        reference_spec1 = self.objective1.reference_spec()
        reference_spec2 = self.objective2.reference_spec()
        assert reference_spec1.type == reference_spec2.type
        shape = (reference_spec1.size + reference_spec2.size,)
        return TensorSpec(type=reference_spec1.type, shape=shape)

    def optimizer_arguments(self, **kwargs):
        arguments = super().optimizer_arguments()
        util.deep_disjoint_update(
            target=arguments, source=self.objective1.optimizer_arguments(**kwargs)
        )
        util.deep_disjoint_update(
            target=arguments, source=self.objective2.optimizer_arguments(**kwargs)
        )
        return arguments

    @tf_function(num_args=5)
    def reference(self, *, states, horizons, internals, auxiliaries, actions, policy):
        reference1 = self.objective1.reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, policy=policy
        )

        reference2 = self.objective2.reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, policy=policy
        )

        shape = (-1, self.objective1.reference_spec().size)
        reference1 = tf.reshape(tensor=reference1, shape=shape)
        shape = (-1, self.objective2.reference_spec().size)
        reference2 = tf.reshape(tensor=reference2, shape=shape)

        return tf.concat(values=(reference1, reference2), axis=1)

    @tf_function(num_args=7)
    def loss(
        self, *, states, horizons, internals, auxiliaries, actions, reward, reference, policy,
        baseline=None
    ):
        reference_spec1 = self.objective1.reference_spec()
        reference_spec2 = self.objective2.reference_spec()
        assert tf_util.shape(x=reference)[1] == reference_spec1.size + reference_spec2.size

        reference1 = reference[:, :reference_spec1.size]
        reference1 = tf.reshape(tensor=reference1, shape=((-1,) + reference_spec1.shape))
        reference2 = reference[:, reference_spec1.size:]
        reference2 = tf.reshape(tensor=reference2, shape=((-1,) + reference_spec2.shape))

        loss1 = self.objective1.loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, reference=reference1, policy=policy, baseline=baseline
        )

        loss2 = self.objective2.loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, reference=reference2, policy=policy, baseline=baseline
        )

        return loss1 + loss2
