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
        parameters_spec (specification): Distribution parameters specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, action_spec, embedding_shape, parameters_spec, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels, l2_regularization=0.0)

        self.action_spec = action_spec
        self.embedding_shape = tuple(embedding_shape)
        self.parameters_spec = parameters_spec

    def input_signature(self, function):
        if function == 'action_value':
            return [
                util.to_tensor_spec(value_spec=self.parameters_spec, batched=True),
                util.to_tensor_spec(value_spec=self.action_spec, batched=True)
            ]

        elif function == 'entropy':
            return [util.to_tensor_spec(value_spec=self.parameters_spec, batched=True)]

        elif function == 'kl_divergence':
            return [
                util.to_tensor_spec(value_spec=self.parameters_spec, batched=True),
                util.to_tensor_spec(value_spec=self.parameters_spec, batched=True)
            ]

        elif function == 'log_probability':
            return [
                util.to_tensor_spec(value_spec=self.parameters_spec, batched=True),
                util.to_tensor_spec(value_spec=self.action_spec, batched=True)
            ]

        elif function == 'parametrize':
            return [util.to_tensor_spec(
                value_spec=dict(type='float', shape=self.embedding_shape), batched=True
            )]

        elif function == 'sample':
            return [
                util.to_tensor_spec(value_spec=self.parameters_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='float', shape=()), batched=False)
            ]

        elif function == 'states_value':
            return [util.to_tensor_spec(value_spec=self.parameters_spec, batched=True)]

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=1)
    def parametrize(self, x):
        raise NotImplementedError

    @tf_function(num_args=2)
    def sample(self, parameters, temperature):
        raise NotImplementedError

    @tf_function(num_args=2)
    def log_probability(self, parameters, action):
        raise NotImplementedError

    @tf_function(num_args=1)
    def entropy(self, parameters):
        raise NotImplementedError

    @tf_function(num_args=2)
    def kl_divergence(self, parameters1, parameters2):
        raise NotImplementedError

    @tf_function(num_args=1)
    def states_value(self, parameters):
        raise NotImplementedError

    @tf_function(num_args=2)
    def action_value(self, parameters, action):
        raise NotImplementedError
