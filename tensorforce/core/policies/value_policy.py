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

from tensorforce.core import SignatureDict, TensorSpec, tf_function
from tensorforce.core.policies import ActionValue, Policy, StateValue


class ValuePolicy(Policy, StateValue, ActionValue):
    """
    Base class for value-based policies.

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

    def __init__(
        self, *, device=None, l2_regularization=None, name=None, states_spec=None,
        auxiliaries_spec=None, actions_spec=None
    ):
        Policy.__init__(
            self=self, device=device, l2_regularization=l2_regularization, name=name,
            states_spec=states_spec, auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec
        )

    def input_signature(self, *, function):
        if function == 'action_value':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True)
            )

        elif function == 'action_values':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True),
                actions=self.actions_spec.signature(batched=True)
            )

        elif function == 'state_values':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        else:
            try:
                return Policy.input_signature(self=self, function=function)
            except NotImplementedError:
                try:
                    return StateValue.input_signature(self=self, function=function)
                except NotImplementedError:
                    return ActionValue.input_signature(self=self, function=function)

    def output_signature(self, *, function):
        if function == 'action_value':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=()).signature(batched=True)
            )

        if function == 'action_values':
            return SignatureDict(
                singleton=self.actions_spec.fmap(function=(
                    lambda spec: TensorSpec(type='float', shape=spec.shape).signature(batched=True)
                ), cls=SignatureDict)
            )

        elif function == 'state_values':
            return SignatureDict(
                singleton=self.actions_spec.fmap(function=(
                    lambda spec: TensorSpec(type='float', shape=spec.shape).signature(batched=True)
                ), cls=SignatureDict)
            )

        else:
            try:
                return Policy.output_signature(self=self, function=function)
            except NotImplementedError:
                try:
                    return StateValue.output_signature(self=self, function=function)
                except NotImplementedError:
                    return ActionValue.output_signature(self=self, function=function)

    @tf_function(num_args=5)
    def action_value(self, *, states, horizons, internals, auxiliaries, actions):
        action_values = self.action_values(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions
        )

        def function(value, spec):
            return tf.reshape(tensor=value, shape=(-1, spec.size))

        action_values = action_values.fmap(function=function, zip_values=self.actions_spec)
        action_values = tf.concat(values=tuple(action_values.values()), axis=1)

        return tf.math.reduce_mean(input_tensor=action_values, axis=1)

    @tf_function(num_args=4)
    def state_value(self, *, states, horizons, internals, auxiliaries):
        state_values = self.state_values(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        )

        def function(value, spec):
            return tf.reshape(tensor=value, shape=(-1, spec.size))

        state_values = state_values.fmap(function=function, zip_values=self.actions_spec)
        state_values = tf.concat(values=tuple(state_values.values()), axis=1)

        return tf.math.reduce_mean(input_tensor=state_values, axis=1)

    @tf_function(num_args=5)
    def action_values(self, *, states, horizons, internals, auxiliaries, actions):
        raise NotImplementedError

    @tf_function(num_args=4)
    def state_values(self, *, states, horizons, internals, auxiliaries):
        raise NotImplementedError
