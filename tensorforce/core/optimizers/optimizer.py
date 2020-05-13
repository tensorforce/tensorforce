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
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, summary_labels=None, name=None, arguments_spec=None):
        super().__init__(name=name, summary_labels=summary_labels)

        self.arguments_spec = arguments_spec

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
        assert all(variable.dtype.is_floating for variable in variables)
        # for variable in variables:
        #     tf.debugging.assert_type(
        #         tensor=variable, tf_type=tf_util.get_dtype(type='float'),
        #         message="Optimizer.update: invalid type for optimized variable {}: {}.".format(
        #             variable.name, variable.dtype
        #         )
        #     )

        deltas = self.step(arguments=arguments, variables=variables, **kwargs)

        update_norm = tf.linalg.global_norm(
            t_list=[tf_util.cast(x=delta, dtype='float') for delta in deltas]
        )
        deltas = self.add_summary(
            label='update-norm', name='update-norm', tensor=update_norm, pass_tensors=deltas
        )

        for n in range(len(variables)):
            name = variables[n].name
            if name[-2:] != ':0':
                raise TensorforceError.unexpected()
            deltas[n] = self.add_summary(
                label='updates', name=('update-' + name[:-2]), tensor=deltas[n], mean_variance=True
            )
            deltas[n] = self.add_summary(
                label='updates-histogram', name=('update-' + name[:-2]), tensor=deltas[n]
            )

        with tf.control_dependencies(control_inputs=deltas):
            return tf_util.identity(input=tf_util.constant(value=True, dtype='bool'))
