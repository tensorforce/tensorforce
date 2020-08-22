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

from tensorforce.core import Module, SignatureDict, TensorSpec, tf_function


class Objective(Module):
    """
    Base class for optimization objectives.

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
        super().__init__(name=name)

        self.states_spec = states_spec
        self.internals_spec = internals_spec
        self.auxiliaries_spec = auxiliaries_spec
        self.actions_spec = actions_spec
        self.reward_spec = reward_spec

    def reference_spec(self):
        raise NotImplementedError

    def required_policy_fns(self):
        return ()

    def required_baseline_fns(self):
        return ()

    def input_signature(self, *, function):
        if function == 'loss':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True),
                reward=self.reward_spec.signature(batched=True),
                reference=self.reference_spec().signature(batched=True)
            )

        elif function == 'reference':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True)
            )

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'loss':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=()).signature(batched=True)
            )

        elif function == 'reference':
            return SignatureDict(singleton=self.reference_spec().signature(batched=True))

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=5)
    def reference(self, *, states, horizons, internals, auxiliaries, actions, policy):
        raise NotImplementedError

    @tf_function(num_args=7)
    def loss(
        self, *, states, horizons, internals, auxiliaries, actions, reward, reference, policy,
        baseline=None
    ):
        raise NotImplementedError
