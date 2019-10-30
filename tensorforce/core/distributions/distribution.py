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


class Distribution(Module):
    """
    Base class for policy distributions.

    Args:
        name (string): Distribution name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        action_spec (specification): Action specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        embedding_shape (iter[int > 0]): Embedding shape
            (<span style="color:#0000C0"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, action_spec, embedding_shape, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels, l2_regularization=0.0)

        self.action_spec = action_spec
        self.embedding_shape = tuple(embedding_shape)

    def tf_parametrize(self, x):
        raise NotImplementedError

    def tf_sample(self, parameters, temperature):
        raise NotImplementedError

    def tf_log_probability(self, parameters, action):
        raise NotImplementedError

    def tf_entropy(self, parameters):
        raise NotImplementedError

    def tf_kl_divergence(self, parameters1, parameters2):
        raise NotImplementedError

    def tf_action_value(self, parameters, action=None):
        raise NotImplementedError

    def tf_states_value(self, parameters):
        raise NotImplementedError
