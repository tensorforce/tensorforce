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

from tensorforce import util
import tensorforce.core
from tensorforce.core import tf_function
from tensorforce.core.objectives import Objective


class Plus(Objective):
    """
    Additive combination of two objectives (specification key: `plus`).

    Args:
        objective1 (specification): First objective configuration
            (<span style="color:#C00000"><b>required</b></span>).
        objective2 (specification): Second objective configuration
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        reward_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, objective1, objective2, summary_labels=None, name=None, states_spec=None,
        internals_spec=None, auxiliaries_spec=None, actions_spec=None, reward_spec=None
    ):
        super().__init__(
            summary_labels=summary_labels, name=name, states_spec=states_spec,
            internals_spec=internals_spec, auxiliaries_spec=auxiliaries_spec,
            actions_spec=actions_spec, reward_spec=reward_spec
        )

        self.objective1 = self.add_module(
            name='objective1', module=objective1, modules=tensorforce.core.objective_modules,
            states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

        self.objective2 = self.add_module(
            name='objective2', module=objective2, modules=tensorforce.core.objective_modules,
            states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

    def reference_spec(self):
        return [self.objective1.reference_spec(), self.objective2.reference_spec()]

    def optimizer_arguments(self, **kwargs):
        arguments = super().optimizer_arguments()
        util.deep_disjoint_update(
            target=arguments, source=self.objective1.optimizer_arguments(**kwargs)
        )
        util.deep_disjoint_update(
            target=arguments, source=self.objective2.optimizer_arguments(**kwargs)
        )
        return arguments

    @tf_function(num_args=6)
    def loss(self, *, states, horizons, internals, auxiliaries, actions, reward, policy):
        loss1 = self.objective1.loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, policy=policy
        )

        loss2 = self.objective2.loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, policy=policy
        )

        return loss1 + loss2

    @tf_function(num_args=6)
    def reference(self, *, states, horizons, internals, auxiliaries, actions, reward, policy):
        reference1 = self.objective1.reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, policy=policy
        )

        reference2 = self.objective2.reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, policy=policy
        )

        return (reference1, reference2)

    @tf_function(num_args=7)
    def comparative_loss(
        self, *, states, horizons, internals, auxiliaries, actions, reward, reference, policy
    ):
        loss1 = self.objective1.comparative_loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, reference=reference[0], policy=policy
        )

        loss2 = self.objective2.comparative_loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, reference=reference[1], policy=policy
        )

        return loss1 + loss2
