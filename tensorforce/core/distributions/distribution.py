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


class Distribution(Module):
    """
    Base class for policy distributions.

    Args:
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        action_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        input_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        parameters_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        conditions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, name=None, action_spec=None, input_spec=None, parameters_spec=None,
        conditions_spec=None
    ):
        assert input_spec.type == 'float'
        super().__init__(l2_regularization=0.0, name=name)

        self.action_spec = action_spec
        self.input_spec = input_spec
        self.parameters_spec = parameters_spec
        self.conditions_spec = conditions_spec

    def input_signature(self, *, function):
        if function == 'action_value':
            return SignatureDict(
                parameters=self.parameters_spec.signature(batched=True),
                action=self.action_spec.signature(batched=True)
            )

        elif function == 'entropy':
            return SignatureDict(parameters=self.parameters_spec.signature(batched=True))

        elif function == 'kl_divergence':
            return SignatureDict(
                parameters1=self.parameters_spec.signature(batched=True),
                parameters2=self.parameters_spec.signature(batched=True)
            )

        elif function == 'log_probability':
            return SignatureDict(
                parameters=self.parameters_spec.signature(batched=True),
                action=self.action_spec.signature(batched=True)
            )

        elif function == 'mode':
            return SignatureDict(parameters=self.parameters_spec.signature(batched=True))

        elif function == 'parametrize':
            return SignatureDict(
                x=self.input_spec.signature(batched=True),
                conditions=self.conditions_spec.signature(batched=True)
            )

        elif function == 'sample':
            return SignatureDict(
                parameters=self.parameters_spec.signature(batched=True),
                temperature=TensorSpec(type='float', shape=()).signature(batched=False)
            )

        elif function == 'state_value':
            return SignatureDict(parameters=self.parameters_spec.signature(batched=True))

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        shape = self.action_spec.shape

        if function == 'action_value':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=shape).signature(batched=True)
            )

        elif function == 'entropy':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=shape).signature(batched=True)
            )

        elif function == 'kl_divergence':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=shape).signature(batched=True)
            )

        elif function == 'log_probability':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=shape).signature(batched=True)
            )

        elif function == 'mode':
            return SignatureDict(singleton=self.action_spec.signature(batched=True))

        elif function == 'parametrize':
            return SignatureDict(singleton=self.parameters_spec.signature(batched=True))

        elif function == 'sample':
            return SignatureDict(singleton=self.action_spec.signature(batched=True))

        elif function == 'state_value':
            return SignatureDict(
                singleton=TensorSpec(type='float', shape=shape).signature(batched=True)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=2)
    def parametrize(self, *, x, conditions):
        raise NotImplementedError

    @tf_function(num_args=1)
    def mode(self, *, parameters):
        raise NotImplementedError

    @tf_function(num_args=2)
    def sample(self, *, parameters, temperature):
        raise NotImplementedError

    @tf_function(num_args=2)
    def log_probability(self, *, parameters, action):
        raise NotImplementedError

    @tf_function(num_args=1)
    def entropy(self, *, parameters):
        raise NotImplementedError

    @tf_function(num_args=2)
    def kl_divergence(self, *, parameters1, parameters2):
        raise NotImplementedError

    @tf_function(num_args=2)
    def action_value(self, *, parameters, action):
        raise NotImplementedError

    @tf_function(num_args=1)
    def state_value(self, *, parameters):
        raise NotImplementedError
