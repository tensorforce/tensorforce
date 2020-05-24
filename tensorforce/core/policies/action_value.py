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
from tensorforce.core import SignatureDict, TensorSpec, tf_function
from tensorforce.core.policies import Policy


class ActionValue(Policy):
    """
    Base class for action-value-based policies.

    Args:
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def input_signature(self, *, function):
        if function == 'actions_value':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True)
            )

        elif function == 'actions_values':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True)
            )

        elif function == 'all_actions_values':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'states_value':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'states_values':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=5)
    def actions_value(
        self, *, states, horizons, internals, auxiliaries, actions, reduced, return_per_action
    ):
        actions_values = self.actions_values(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions
        )

        return self.join_value_per_action(
            values=actions_values, reduced=reduced, return_per_action=return_per_action
        )

    @tf_function(num_args=5)
    def actions_values(self, *, states, horizons, internals, auxiliaries, actions):
        raise NotImplementedError

    @tf_function(num_args=4)
    def all_actions_values(self, *, states, horizons, internals, auxiliaries):
        raise NotImplementedError

    @tf_function(num_args=4)
    def states_value(self, *, states, horizons, internals, auxiliaries, reduced, return_per_action):
        states_values = self.states_values(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        )

        return self.join_value_per_action(
            values=states_values, reduced=reduced, return_per_action=return_per_action
        )

    @tf_function(num_args=4)
    def states_values(self, *, states, horizons, internals, auxiliaries):
        actions_values = self.all_actions_values(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        )

        return actions_values.fmap(
            function=(lambda action_values: tf.math.reduce_max(input_tensor=action_values, axis=-1))
        )
