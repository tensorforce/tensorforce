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


class Memory(Module):
    """
    Base class for memories.

    Args:
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        values_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        min_capacity (int >= 0): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, device=None, summary_labels=None, l2_regularization=None, name=None,
        values_spec=None, min_capacity=None
    ):
        super().__init__(
            name=name, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.values_spec = values_spec
        self.min_capacity = min_capacity

    def input_signature(self, *, function):
        if function == 'enqueue':
            return self.values_spec.signature(batched=True)

        elif function == 'predecessors':
            return SignatureDict(
                indices=TensorSpec(type='int', shape=()).signature(batched=True),
                horizon=TensorSpec(type='int', shape=()).signature(batched=False)
            )

        elif function == 'retrieve':
            return SignatureDict(indices=TensorSpec(type='int', shape=()).signature(batched=True))

        elif function == 'retrieve_episodes':
            return SignatureDict(n=TensorSpec(type='int', shape=()).signature(batched=False))

        elif function == 'retrieve_timesteps':
            return SignatureDict(
                n=TensorSpec(type='int', shape=()).signature(batched=False),
                past_horizon=TensorSpec(type='int', shape=()).signature(batched=False),
                future_horizon=TensorSpec(type='int', shape=()).signature(batched=False)
            )

        elif function == 'successors':
            return SignatureDict(
                indices=TensorSpec(type='int', shape=()).signature(batched=True),
                horizon=TensorSpec(type='int', shape=()).signature(batched=False)
            )

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=6)
    def enqueue(self, *, states, internals, auxiliaries, actions, terminal, reward):
        raise NotImplementedError

    @tf_function(num_args=1)
    def retrieve(self, *, indices, values):
        raise NotImplementedError

    @tf_function(num_args=2)
    def successors(self, *, indices, horizon, sequence_values, final_values):
        raise NotImplementedError

    @tf_function(num_args=2)
    def predecessors(self, *, indices, horizon, sequence_values, initial_values):
        raise NotImplementedError

    @tf_function(num_args=3)
    def retrieve_timesteps(self, *, n, past_horizon, future_horizon):
        raise NotImplementedError

    @tf_function(num_args=1)
    def retrieve_episodes(self, *, n):
        raise NotImplementedError
