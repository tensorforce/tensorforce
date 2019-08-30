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
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def tf_enqueue(self, states, internals, actions, terminal, reward):
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
