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
from tensorforce.core import Module, tf_function


class Objective(Module):
    """
    Base class for optimization objectives.

    Args:
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, summary_labels=None, name=None, states_spec=None, internals_spec=None,
        auxiliaries_spec=None, actions_spec=None
    ):
        super().__init__(name=name, summary_labels=summary_labels)

        self.states_spec = states_spec
        self.internals_spec = internals_spec
        self.auxiliaries_spec = auxiliaries_spec
        self.actions_spec = actions_spec

    def reference_spec(self):
        return dict(type='float', shape=())

    def input_signature(self, function):
        if function == 'loss':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec, batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                util.to_tensor_spec(value_spec=self.actions_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='float', shape=()), batched=True)
            ]

        elif function == 'reference':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec, batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                util.to_tensor_spec(value_spec=self.actions_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='float', shape=()), batched=True)
            ]

        elif function == 'comparative_loss':
            return [
                util.to_tensor_spec(value_spec=self.states_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(value_spec=self.internals_spec, batched=True),
                util.to_tensor_spec(value_spec=self.auxiliaries_spec, batched=True),
                util.to_tensor_spec(value_spec=self.actions_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='float', shape=()), batched=True),
                util.to_tensor_spec(value_spec=self.reference_spec(), batched=True)
            ]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=6)
    def loss(self, states, horizons, internals, auxiliaries, actions, reward, policy):
        reference = self.reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, policy=policy
        )
        return self.comparative_loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, reference=reference, policy=policy
        )

    @tf_function(num_args=6)
    def reference(self, states, horizons, internals, auxiliaries, actions, reward, policy):
        return tf.zeros_like(input=reward)

    @tf_function(num_args=7)
    def comparative_loss(
        self, states, horizons, internals, auxiliaries, actions, reward, reference, policy
    ):
        return self.loss(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, policy=policy
        )

    def optimizer_arguments(self, **kwargs):
        return OrderedDict()
