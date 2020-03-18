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

from tensorforce import util
from tensorforce.core import Module, tf_function


class Objective(Module):
    """
    Base class for optimization objectives.

    Args:
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, summary_labels=None, name=None, policy=None):
        super().__init__(name=name, summary_labels=summary_labels)

    def input_signature(self, function):
        if function == 'loss_per_instance':
            return [
                util.to_tensor_spec(value_spec=self.parent.states_spec, batched=True),
                util.to_tensor_spec(value_spec=self.parent.internals_spec, batched=True),
                util.to_tensor_spec(value_spec=self.parent.auxiliaries_spec, batched=True),
                util.to_tensor_spec(value_spec=self.parent.actions_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='float', shape=()), batched=True)
            ]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=5)
    def loss_per_instance(self, states, internals, auxiliaries, actions, reward):
        raise NotImplementedError

    def optimizer_arguments(self, **kwargs):
        return OrderedDict()
