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

from tensorforce.core import Module, SignatureDict, TensorSpec, TensorsSpec, tf_function, tf_util


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

        name = self.name[:self.name.index('_')] + '-update/norm'
        self.register_summary(label='update-norm', name=name)

    def initialize_given_variables(self, *, variables, register_summaries):
        assert not self.root.is_initialized and not self.is_initialized_given_variables

        for module in self.this_submodules:
            if isinstance(module, Optimizer):
                module.initialize_given_variables(variables=variables, register_summaries=False)

        # Replace "/" with "_" to ensure TensorDict is flat
        self.variables_spec = TensorsSpec(((var.name[:-2].replace('/', '_'), TensorSpec(
            type=tf_util.dtype(x=var, fallback_tf_dtype=True), shape=tf_util.shape(x=var)
        )) for var in variables))

        if register_summaries:
            assert self.is_initialized
            self.is_initialized = False
            prefix = self.name[:self.name.index('_')] + '-updates/'
            names = list()
            for variable in variables:
                assert variable.name.startswith(self.root.name + '/') and variable.name[-2:] == ':0'
                names.append(prefix + variable.name[len(self.root.name) + 1: -2] + '-mean')
                names.append(prefix + variable.name[len(self.root.name) + 1: -2] + '-variance')
            self.register_summary(label='updates', name=names)
            self.is_initialized = True

        self.is_initialized_given_variables = True

    def input_signature(self, *, function):
        if function == 'step' or function == 'update':
            return SignatureDict(arguments=self.arguments_spec.signature(batched=True))

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'step':
            return self.variables_spec.fmap(
                function=(lambda spec: spec.signature(batched=True)), cls=SignatureDict
            )

        elif function == 'update':
            return SignatureDict(
                singleton=TensorSpec(type='bool', shape=()).signature(batched=False)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        raise NotImplementedError

    @tf_function(num_args=1)
    def update(self, *, arguments, variables, **kwargs):
        assert self.is_initialized_given_variables
        assert all(variable.dtype.is_floating for variable in variables)

        deltas = self.step(arguments=arguments, variables=variables, **kwargs)
        dependencies = list(deltas)

        def fn_summary():
            return tf.linalg.global_norm(
                t_list=[tf_util.cast(x=delta, dtype='float') for delta in deltas]
            )

        assertions = list(deltas)
        # if self.config.create_debug_assertions:
        #     from tensorforce.core.optimizers import Synchronization
        #     if not isinstance(self, Synchronization) or \
        #             not self.sync_frequency.is_constant(value=1):
        #         for delta, variable in zip(deltas, variables):
        #             if variable.shape.num_elements() <= 4:
        #                 continue
        #             # if '_distribution/mean/linear/' in variable.name:
        #             #     continue
        #             assertions.append(tf.debugging.assert_equal(
        #                 x=tf.math.reduce_any(
        #                     input_tensor=tf.math.not_equal(x=delta, y=tf.zeros_like(input=delta))
        #                 ), y=tf_util.constant(value=True, dtype='bool'), message=variable.name
        #             ))

        name = self.name[:self.name.index('_')] + '-update/norm'
        dependencies.extend(
            self.summary(label='update-norm', name=name, data=fn_summary, step='updates')
        )

        with tf.control_dependencies(control_inputs=assertions):

            def fn_summary():
                xs = list()
                for variable in variables:
                    xs.extend(tf.nn.moments(x=variable, axes=list(range(tf_util.rank(x=variable)))))
                return xs

            prefix = self.name[:self.name.index('_')] + '-updates/'
            names = list()
            for variable in variables:
                assert variable.name.startswith(self.root.name + '/') and variable.name[-2:] == ':0'
                names.append(prefix + variable.name[len(self.root.name) + 1: -2] + '-mean')
                names.append(prefix + variable.name[len(self.root.name) + 1: -2] + '-variance')
            dependencies.extend(
                self.summary(label='updates', name=names, data=fn_summary, step='updates')
            )

        with tf.control_dependencies(control_inputs=dependencies):
            return tf_util.identity(input=tf_util.constant(value=True, dtype='bool'))
