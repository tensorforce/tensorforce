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
from tensorforce.core import Module, SignatureDict, TensorDict, TensorSpec, TensorsSpec, tf_function


class Policy(Module):
    """
    Base class for decision policies.

    Args:
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, device=None, summary_labels=None, l2_regularization=None, name=None,
        states_spec=None, auxiliaries_spec=None, actions_spec=None
    ):
        super().__init__(
            device=device, summary_labels=summary_labels, l2_regularization=l2_regularization,
            name=name
        )

        self.states_spec = states_spec
        self.auxiliaries_spec = auxiliaries_spec
        self.actions_spec = actions_spec

    @property
    def internals_spec(self):
        return TensorsSpec()

    def internals_init(self):
        return TensorDict()

    def max_past_horizon(self, *, on_policy):
        raise NotImplementedError

    def input_signature(self, *, function):
        if function == 'act':
            return SignatureDict(
                states=self.states_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                auxiliaries=self.auxiliaries_spec.signature(batched=True)
            )

        elif function == 'join_value_per_action':
            return SignatureDict(
                values=self.actions_spec.fmap(
                    function=(lambda x: TensorSpec(type='float', shape=x.shape))
                ).signature(batched=True)
            )

        elif function == 'past_horizon':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=0)
    def past_horizon(self, *, on_policy):
        raise NotImplementedError

    @tf_function(num_args=4)
    def act(self, *, states, horizons, internals, auxiliaries, deterministic, return_internals):
        raise NotImplementedError

    @tf_function(num_args=1)
    def join_value_per_action(self, *, values, reduced, return_per_action):
        assert not return_per_action or reduced

        def function(value, spec):
            return tf.reshape(tensor=value, shape=(-1, util.product(xs=spec.shape)))

        values = values.fmap(function=function, zip_values=self.actions_spec)

        value = tf.concat(values=tuple(values.values()), axis=1)

        if reduced:
            value = tf.math.reduce_mean(input_tensor=value, axis=1)

            if return_per_action:
                values = values.fmap(
                    function=(lambda x: tf.math.reduce_mean(input_tensor=x, axis=1))
                )
                return value, values

            else:
                return value

        else:
            return value
