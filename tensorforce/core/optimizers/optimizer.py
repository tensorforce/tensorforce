# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce import util, TensorForceError
import tensorforce.core.optimizers


class Optimizer(tf.train.GradientDescentOptimizer):
    """
    Generic TensorFlow optimizer which minimizes a not yet further specified expression, usually  
    some kind of loss function. More generally, an optimizer can be considered as some method of  
    *updating* a set of variables.
    """

    def __init__(self):
        """
        Creates a new optimizer instance.
        """
        super(Optimizer, self).__init__(self._learning_rate, use_locking=False, name='TensorForceOptimizer')

        self._learning_rate = -1.0

        self.variables = dict()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            assert kwargs.get('trainable', False)
            if not registered:
                self.variables[name] = variable
            return variable

        # TensorFlow function
        self.step = tf.make_template(
            name_='step',
            func_=self.tf_step,
            custom_getter=custom_getter
        )

    def tf_step(self, time, variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            **kwargs: Additional arguments depending on the specific optimizer implementation.  
            For instance, often includes `fn_loss` if a loss function is optimized.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        raise NotImplementedError

    def minimize(self, time, variables, **kwargs):
        """
        Performs an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            **kwargs: Additional optimizer-specific arguments. For instance, often includes  
            `fn_loss` if a loss function is optimized.

        Returns:
            The optimization operation.
        """
        deltas = self.step(time=time, variables=variables, **kwargs)
        # deltas[0] = tf.Print(deltas[0], (deltas[0],))
        with tf.control_dependencies(control_inputs=deltas):
            return tf.no_op()

    def get_variables(self):
        """
        Returns the TensorFlow variables used by the optimizer.

        Returns:
            List of variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates an optimizer from a specification dict.
        """
        optimizer = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.optimizers.optimizers,
            kwargs=kwargs
        )
        assert isinstance(optimizer, Optimizer)
        return optimizer

    # modified minimize
    def apply_step(
        self,
        variables,
        deltas,
        global_step=None,
        gate_gradients=None,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        name=None,
        grad_loss=None
    ):

        deltas_and_vars = self.compute_deltas(
            deltas=deltas,
            var_list=variables,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss
        )

        vars_with_delta = [v for g, v in deltas_and_vars if g is not None]
        if not vars_with_delta:
            raise TensorForceError(
                "No gradients provided for any variable, check your graph for ops that do not "
                "support gradients, between variables {} and loss {}".format(
                    [str(v) for _, v in deltas_and_vars], deltas
                )
            )

        return super(Optimizer, self).apply_gradients(deltas_and_vars, global_step=global_step, name=name)

    def compute_gradients(self, *args, **kwargs):
        raise NotImplementedError

    def apply_gradients(self, *args, **kwargs):
        raise NotImplementedError

    # Modified compute_gradients
    def compute_deltas(
        self,
        deltas,
        var_list=None,
        gate_gradients=None,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None
    ):
        if aggregation_method is not None or colocate_gradients_with_ops or grad_loss is not None:
            raise TensorForceError("'aggregation_method', colocate_gradients_with_ops' and 'grad_loss' arguments are not supported.")
        if gate_gradients is None:
            gate_gradients = Optimizer.GATE_OP
        if gate_gradients not in (Optimizer.GATE_NONE, Optimizer.GATE_OP, Optimizer.GATE_GRAPH):
            raise TensorForceError("'gate_gradients' must be one of: Optimizer.GATE_NONE, Optimizer.GATE_OP, Optimizer.GATE_GRAPH. Not {}".format(gate_gradients))
        # if isinstance(loss, tf.Tensor):
        #     self._assert_valid_dtypes([loss])
        # else:
        #     self._assert_valid_dtypes(loss)
        # if var_list is None:
        #     var_list = tf.trainable_variables() + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        # else:
        #     var_list = tf.python.util.nest.flatten(var_list)
        var_list += tf.get_collection(tf.GraphKeys._STREAMING_MODEL_PORTS)
        if not var_list:
            raise TensorForceError("No variables to optimize.")
        # processors = [tf.train.Optimizer._get_processor(v) for v in var_list]
        # var_refs = [p.target() for p in processors]
        # grads = gradients.gradients(loss, var_refs, grad_ys=grad_loss, gate_gradients=(gate_gradients == Optimizer.GATE_OP), aggregation_method=aggregation_method, colocate_gradients_with_ops=colocate_gradients_with_ops)

        if gate_gradients == Optimizer.GATE_GRAPH:
            deltas = tf.tuple(deltas)
        deltas_and_vars = list(zip(deltas, var_list))
        self._assert_valid_dtypes([v for g, v in deltas_and_vars if g is not None and v.dtype != tf.resource])
        return deltas_and_vars

    # Below, we just pass through tf optimizers
    def _prepare(self):
        return tf.train.GradientDescentOptimizer._prepare(self=self)

    def _apply_dense(self, grad, var):
        return tf.train.GradientDescentOptimizer._apply_dense(self=self, grad=grad, var=var)

    def _apply_sparse_duplicate_indices(self, grad, var):
        return tf.train.GradientDescentOptimizer._apply_sparse_duplicate_indices(
            self=self, grad=grad, var=var
        )

    def _resource_apply_dense(self, grad, handle):
        return tf.train.GradientDescentOptimizer._resource_apply_dense(
            self=self, grad=grad, handle=handle
        )

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return tf.train.GradientDescentOptimizer._resource_apply_sparse_duplicate_indices(
            self=self, grad=grad, handle=handle
        )
