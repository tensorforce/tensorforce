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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import Module


class Optimizer(Module):
    """
    Base class for optimizers.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels)

    def tf_step(self, variables, **kwargs):
        raise NotImplementedError

    def tf_apply_step(self, variables, deltas):
        if len(variables) != len(deltas):
            raise TensorforceError("Invalid variables and deltas lists.")

        assignments = list()
        for variable, delta in zip(variables, deltas):
            assignments.append(variable.assign_add(delta=delta, read_value=False))

        with tf.control_dependencies(control_inputs=assignments):
            return util.no_operation()

    def tf_minimize(self, variables, **kwargs):
        if any(variable.dtype != util.tf_dtype(dtype='float') for variable in variables):
            TensorforceError.unexpected()

        deltas = self.step(variables=variables, **kwargs)

        update_norm = tf.linalg.global_norm(t_list=deltas)
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

        # TODO: experimental
        # with tf.control_dependencies(control_inputs=deltas):
        #     zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        #     false = tf.constant(value=False, dtype=util.tf_dtype(dtype='bool'))
        #     deltas = [self.cond(
        #         pred=tf.math.reduce_all(input_tensor=tf.math.equal(x=delta, y=zero)),
        #         true_fn=(lambda: tf.Print(delta, (variable.name,))),
        #         false_fn=(lambda: delta)) for delta, variable in zip(deltas, variables)
        #     ]
        #     assertions = [
        #         tf.debugging.assert_equal(
        #             x=tf.math.reduce_all(input_tensor=tf.math.equal(x=delta, y=zero)), y=false
        #         ) for delta, variable in zip(deltas, variables)
        #         if util.product(xs=util.shape(x=delta)) > 4 and 'distribution' not in variable.name
        #     ]

        # with tf.control_dependencies(control_inputs=assertions):
        with tf.control_dependencies(control_inputs=deltas):
            return util.no_operation()

    def add_variable(self, name, dtype, shape, is_trainable=False, initializer='zeros'):
        if is_trainable:
            raise TensorforceError("Invalid trainable variable.")

        return super().add_variable(
            name=name, dtype=dtype, shape=shape, is_trainable=is_trainable, initializer=initializer
        )
