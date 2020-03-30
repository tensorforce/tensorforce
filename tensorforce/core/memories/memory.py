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

from tensorforce import util
from tensorforce.core import Module, tf_function


class Memory(Module):
    """
    Base class for memories.

    Args:
        name (string): Memory name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        values_spec (specification): Values specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        min_capacity (int >= 0): Minimum memory capacity
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """
    def __init__(
        self, name, values_spec, min_capacity=0, device=None, summary_labels=None,
        l2_regularization=None
    ):
        super().__init__(
            name=name, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.values_spec = values_spec


    def input_signature(self, function):
        if function == 'action_value':
            return [
                util.to_tensor_spec(value_spec=self.parameters_spec, batched=True),
                util.to_tensor_spec(value_spec=self.action_spec, batched=True)
            ]

    def input_signature(self, function):
        if function == 'enqueue':
            return [
                util.to_tensor_spec(value_spec=self.values_spec['states'], batched=True),
                util.to_tensor_spec(value_spec=self.values_spec['internals'], batched=True),
                util.to_tensor_spec(value_spec=self.values_spec['auxiliaries'], batched=True),
                util.to_tensor_spec(value_spec=self.values_spec['actions'], batched=True),
                util.to_tensor_spec(value_spec=self.values_spec['terminal'], batched=True),
                util.to_tensor_spec(value_spec=self.values_spec['reward'], batched=True)
            ]

        elif function == 'predecessors':
            return [
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=False)
            ]

        elif function == 'retrieve':
            return [util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=True)]

        elif function == 'retrieve_episodes':
            return [util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=False)]

        elif function == 'retrieve_timesteps':
            return [
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=False),
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=False),
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=False)
            ]

        elif function == 'successors':
            return [
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=()), batched=False)
            ]

        else:
            assert False

    @tf_function(num_args=6)
    def enqueue(self, states, internals, auxiliaries, actions, terminal, reward):
        raise NotImplementedError

    @tf_function(num_args=1)
    def retrieve(self, indices, values):
        raise NotImplementedError

    @tf_function(num_args=2)
    def successors(self, indices, horizon, sequence_values, final_values):
        raise NotImplementedError

    @tf_function(num_args=2)
    def predecessors(self, indices, horizon, sequence_values, initial_values):
        raise NotImplementedError

    @tf_function(num_args=3)
    def retrieve_timesteps(self, n, past_padding, future_padding):
        raise NotImplementedError

    @tf_function(num_args=1)
    def retrieve_episodes(self, n):
        raise NotImplementedError
