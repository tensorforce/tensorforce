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
from tensorforce.core.policies import Policy


class ActionValue(Policy):
    """
    Base class for action-value-based policies.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        states_spec (specification): States specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        actions_spec (specification): Actions specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    # if any(spec['type'] not in ('bool', 'int') for spec in actions_spec.values()):
    #     raise TensorforceError.unexpected()

    def tf_act(self, states, internals, auxiliaries, return_internals):
        assert return_internals

        actions_values = self.actions_values(
            states=states, internals=internals, auxiliaries=auxiliaries
        )

        actions = OrderedDict()
        for name, spec, action_values in util.zip_items(self.actions_spec, actions_values):
            actions[name] = tf.math.argmax(
                input=action_values, axis=-1, output_type=util.tf_dtype(spec['type'])
            )

        return actions

    def tf_actions_value(
        self, states, internals, auxiliaries, actions, reduced=True, include_per_action=False
    ):
        actions_values = self.actions_values(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions
        )

        for name, spec, actions_value in util.zip_items(self.actions_spec, actions_values):
            actions_values[name] = tf.reshape(
                tensor=actions_value, shape=(-1, util.product(xs=spec['shape']))
            )

        actions_value = tf.concat(values=tuple(actions_values.values()), axis=1)
        if reduced:
            actions_value = tf.math.reduce_mean(input_tensor=actions_value, axis=1)
            if include_per_action:
                for name in self.actions_spec:
                    actions_values[name] = tf.math.reduce_mean(
                        input_tensor=actions_values[name], axis=1
                    )

        if include_per_action:
            actions_values['*'] = actions_value
            return actions_values
        else:
            return actions_value

    def tf_states_value(
        self, states, internals, auxiliaries, reduced=True, include_per_action=False
    ):
        states_values = self.states_values(
            states=states, internals=internals, auxiliaries=auxiliaries
        )

        for name, spec, states_value in util.zip_items(self.actions_spec, states_values):
            states_values[name] = tf.reshape(
                tensor=states_value, shape=(-1, util.product(xs=spec['shape']))
            )

        states_value = tf.concat(values=tuple(states_values.values()), axis=1)
        if reduced:
            states_value = tf.math.reduce_mean(input_tensor=states_value, axis=1)
            if include_per_action:
                for name in self.actions_spec:
                    states_values[name] = tf.math.reduce_mean(
                        input_tensor=states_values[name], axis=1
                    )

        if include_per_action:
            states_values['*'] = states_value
            return states_values
        else:
            return states_value

    def tf_states_values(self, states, internals, auxiliaries):
        if not all(spec['type'] in ('bool', 'int') for spec in self.states_spec.values()):
            raise NotImplementedError

        actions_values = self.actions_values(
            states=states, internals=internals, auxiliaries=auxiliaries
        )

        states_values = OrderedDict()
        for name, spec, action_values in util.zip_items(self.actions_spec, actions_values):
            states_values[name] = tf.math.reduce_max(input_tensor=action_values, axis=-1)

        return states_values

    def tf_actions_values(self, states, internals, auxiliaries, actions=None):
        raise NotImplementedError
