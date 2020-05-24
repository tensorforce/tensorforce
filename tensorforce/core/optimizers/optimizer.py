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

from tensorforce import TensorforceError
from tensorforce.core import Module, SignatureDict, tf_function, tf_util


class Optimizer(Module):
    """
    Base class for optimizers.

    Args:
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, name=None, arguments_spec=None):
        super().__init__(name=name)

        self.arguments_spec = arguments_spec

        self.is_initialized_given_variables = False

    def initialize(self):
        super().initialize()

        self.register_summary(label='update-norm', name='update-norm')

    def initialize_given_variables(self, variables):
        assert not self.root.is_initialized and not self.is_initialized_given_variables
        self.is_initialized_given_variables = True

        assert self.is_initialized
        self.is_initialized = False
        for variable in variables:
            assert variable.name.startswith(self.root.name + '/') and variable.name[-2:] == ':0'
            prefix = 'updates/' + variable.name[len(self.root.name) + 1: -2]
            self.register_summary(label='updates', name=(prefix + '-mean'))
            self.register_summary(label='updates', name=(prefix + '-variance'))
        self.is_initialized = True

    def input_signature(self, *, function):
        if function == 'step' or function == 'update':
            return SignatureDict(arguments=self.arguments_spec.signature(batched=True))

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        raise NotImplementedError

    @tf_function(num_args=1)
    def update(self, *, arguments, variables, **kwargs):
        assert self.is_initialized_given_variables
        assert all(variable.dtype.is_floating for variable in variables)

        deltas = self.step(arguments=arguments, variables=variables, **kwargs)

        update_norm = tf.linalg.global_norm(
            t_list=[tf_util.cast(x=delta, dtype='float') for delta in deltas]
        )
        self.summary(label='update-norm', name='update-norm', data=update_norm, step='updates')

        for variable in variables:
            assert variable.name.startswith(self.root.name + '/') and variable.name[-2:] == ':0'
            mean, var = tf.nn.moments(x=variable, axes=tuple(range(tf_util.rank(x=variable))))
            prefix = 'updates/' + variable.name[len(self.root.name) + 1: -2]
            self.summary(label='updates', name=(prefix + '-mean'), data=mean, step='updates')
            self.summary(label='updates', name=(prefix + '-variance'), data=var, step='updates')

        with tf.control_dependencies(control_inputs=deltas):
            return tf_util.identity(input=tf_util.constant(value=True, dtype='bool'))
