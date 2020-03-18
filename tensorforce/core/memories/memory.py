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

from tensorforce.core import Module


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

    # def __init__(self, name, values_spec, device=None, summary_labels=None, l2_regularization=None):
    #     super().__init__(
    #         name=name, capacity=capacity, values_spec=values_spec, initializers=initializers,
    #         device=device, summary_labels=summary_labels, l2_regularization=l2_regularization
    #     )

    #     self.values_spec = values_spec

    # def input_signature(self, function):
    #     if function == 'enqueue':
    #         return [
    #             [util.to_tensor_spec(value_spec=x) for x in self.values_spec['states'].values()],
    #             [util.to_tensor_spec(value_spec=x) for x in self.values_spec['internals'].values()],
    #             [util.to_tensor_spec(value_spec=x) for x in self.values_spec['auxiliaries'].values()],
    #             [util.to_tensor_spec(value_spec=x) for x in self.values_spec['actions'].values()],
    #             util.to_tensor_spec(value_spec=self.values_spec['terminal']),
    #             util.to_tensor_spec(value_spec=self.values_spec['reward'])
    #         ]

    #     elif function == 'retrieve':
    #         return [util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'))]

    #     elif function == 'successors':
    #         return [
    #             util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'))
    #             util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'), batched=False),
    #         ]

    #     elif function == 'predecessors':
    #         return [
    #             util.to_tensor_spec(value_spec=dict(shape=(), dtype='long')),
    #             util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'), batched=False)
    #         ]

    #     elif function == 'retrieve_timestemps':
    #         return [
    #             util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'), batched=False),
    #             util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'), batched=False),
    #             util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'), batched=False)
    #         ]

    #     elif function == 'retrieve_episodes':
    #         return [util.to_tensor_spec(value_spec=dict(shape=(), dtype='long'), batched=False)]

    #     else:
    #         assert False

    def tf_enqueue(self, states, internals, auxiliaries, actions, terminal, reward):
        raise NotImplementedError

    def tf_retrieve(self, indices, values=None):
        raise NotImplementedError

    def tf_successors(self, indices, horizon, sequence_values=(), final_values=()):
        raise NotImplementedError

    def tf_predecessors(self, indices, horizon, sequence_values=(), initial_values=()):
        raise NotImplementedError

    def tf_retrieve_timesteps(self, n, past_padding, future_padding):
        raise NotImplementedError

    def tf_retrieve_episodes(self, n):
        raise NotImplementedError
