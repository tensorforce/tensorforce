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

    def initialize_given_variables(self, *, variables):
        assert not self.root.is_initialized and not self.is_initialized_given_variables

        for module in self.this_submodules:
            if isinstance(module, Optimizer):
                module.initialize_given_variables(variables=variables)

        # Replace "/" with "_" to ensure TensorDict is flat
        self.variables_spec = TensorsSpec(((var.name[:-2].replace('/', '_'), TensorSpec(
            type=tf_util.dtype(x=var, fallback_tf_dtype=True), shape=tf_util.shape(x=var)
        )) for var in variables))

        self.is_initialized_given_variables = True

    def input_signature(self, *, function):
        if function == 'step' or function == 'update':
            return SignatureDict(arguments=self.arguments_spec.signature(batched=True))

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'step':
            return self.variables_spec.fmap(
                function=(lambda spec: spec.signature(batched=False)), cls=SignatureDict
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

        assertions = list(deltas)
        if self.config.create_debug_assertions:
            from tensorforce.core.optimizers import DoublecheckStep, NaturalGradient, \
                Synchronization, UpdateModifier
            optimizer = self
            while isinstance(optimizer, UpdateModifier):
                if isinstance(optimizer, DoublecheckStep):
                    break
                optimizer = optimizer.optimizer
            if not isinstance(optimizer, DoublecheckStep) and (
                not isinstance(optimizer, NaturalGradient) or not optimizer.only_positive_updates
            ) and (not isinstance(self, Synchronization) or self.sync_frequency is None):
                for delta, variable in zip(deltas, variables):
                    if '_distribution/mean/linear/' in variable.name:
                        # Gaussian.state_value does not use mean
                        continue
                    # if variable.name.endswith('/bias:0') and isinstance(self, Synchronization) \
                    #         and self.root.updates.numpy() == 0:
                    #     # Initialization values are equivalent for bias
                    #     continue
                    assertions.append(tf.debugging.assert_equal(x=tf.math.logical_or(
                        x=tf.math.reduce_all(input_tensor=tf.math.greater(
                            x=tf.math.count_nonzero(input=delta, dtype=tf_util.get_dtype(type='int')),
                            y=tf_util.constant(value=0, dtype='int')
                        )), y=tf.reduce_all(input_tensor=tf.math.equal(
                            x=arguments['reward'], y=tf_util.constant(value=0.0, dtype='float')
                        ))), y=tf_util.constant(value=True, dtype='bool'), message=variable.name
                    ))

        with tf.control_dependencies(control_inputs=assertions):
            dependencies = list()

            if self.root.summaries == 'all' or 'update-norm' in self.root.summaries:
                with self.root.summarizer.as_default():
                    x = tf.linalg.global_norm(
                        t_list=[tf_util.cast(x=delta, dtype='float') for delta in deltas]
                    )
                    dependencies.append(
                        tf.summary.scalar(name='update-norm', data=x, step=self.root.updates)
                    )

            if self.root.summaries == 'all' or 'updates' in self.root.summaries:
                with self.root.summarizer.as_default():
                    for var in variables:
                        assert var.name.startswith(self.root.name + '/') and var.name[-2:] == ':0'
                        mean_name = var.name[len(self.root.name) + 1: -2] + '-mean'
                        var_name = var.name[len(self.root.name) + 1: -2] + '-variance'
                        mean, variance = tf.nn.moments(x=var, axes=list(range(tf_util.rank(x=var))))
                        dependencies.append(
                            tf.summary.scalar(name=mean_name, data=mean, step=self.root.updates)
                        )
                        dependencies.append(
                            tf.summary.scalar(name=var_name, data=variance, step=self.root.updates)
                        )

        with tf.control_dependencies(control_inputs=dependencies):
            return tf_util.identity(input=tf_util.constant(value=True, dtype='bool'))
