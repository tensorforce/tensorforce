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

##TODO - Optimisation package is work in progress
class Optimizer(tf.train.Optimizer):
    """
    Generic optimizer extending the tf.train.Optimizer class. This is for the purpose of having
    a consistent way of handling different types of optimisation such as SGD/momentum variants
    and natural gradients or even evolutionary methods since TensorFlow currently does not offer
    natural gradient optimisers natively.
    """

    def __init__(self, learning_rate=1.0):
        super(Optimizer, self).__init__(use_locking=False, name='TensorForceOptimizer')
        self.learning_rate = self._learning_rate = learning_rate

    def compute_updates(self, variables, loss):
        raise NotImplementedError

    @staticmethod
    def from_config(config, kwargs=None):
        """
        Creates an optimizer from a configuration object.

        Args:
            config: Name of optimizer
            kwargs: Dict of optimizer hyperparameters

        Returns:

        """
        return util.get_object(
            obj=config,
            predefined=tensorforce.core.optimizers.optimizers,
            kwargs=kwargs
        )

    # Inherited tf.train.Optimizer methods, mostly calling the
    # tf.train.GradientDescentOptimizer implementations.

    def compute_gradients(self, loss, var_list=None, gate_gradients=None, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None):
        """
        Computes lists of gradients and variables.

        Args:
            loss:
            var_list:
            gate_gradients:
            aggregation_method:
            colocate_gradients_with_ops:
            grad_loss:

        Returns:

        """
        if aggregation_method is not None or colocate_gradients_with_ops or grad_loss is not None:
            raise TensorForceError("'aggregation_method', colocate_gradients_with_ops' and 'grad_loss' arguments are not supported.")
        if gate_gradients is None:
            gate_gradients = Optimizer.GATE_OP
        if gate_gradients not in (Optimizer.GATE_NONE, Optimizer.GATE_OP, Optimizer.GATE_GRAPH):
            raise TensorForceError("gate_gradients must be one of: Optimizer.GATE_NONE, Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not {}".format(gate_gradients))
        if isinstance(loss, tf.Tensor):
            self._assert_valid_dtypes([loss])
        else:
            self._assert_valid_dtypes(loss)
        if var_list is None:
            var_list = tf.trainable_variables() + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        else:
            var_list = tf.python.util.nest.flatten(var_list)
        var_list += tf.get_collection(tf.GraphKeys._STREAMING_MODEL_PORTS)
        if not var_list:
            raise TensorForceError("No variables to optimize.")
        # processors = [tf.train.Optimizer._get_processor(v) for v in var_list]
        # var_refs = [p.target() for p in processors]
        if isinstance(loss, tf.Tensor):
            grads = self.compute_updates(variables=var_list, loss=loss)
        else:
            grads = self.compute_updates(variables=var_list, **loss)
        if gate_gradients == Optimizer.GATE_GRAPH:
            grads = tf.tuple(grads)
        grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None and v.dtype != tf.resource])

        return grads_and_vars

    def _prepare(self):
        return tf.train.GradientDescentOptimizer._prepare(self=self)
        # self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
        #                                                    name="learning_rate")

    def _apply_dense(self, grad, var):

        return tf.train.GradientDescentOptimizer._apply_dense(self=self, grad=grad, var=var)
        # return training_ops.apply_gradient_descent(
        #     var,
        #     math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        #     grad,
        #     use_locking=self._use_locking).op

    def _apply_sparse_duplicate_indices(self, grad, var):
        return tf.train.GradientDescentOptimizer._apply_sparse_duplicate_indices(self=self, grad=grad, var=var)
        # delta = ops.IndexedSlices(
        #     grad.values *
        #     math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        #     grad.indices, grad.dense_shape)
        # return var.scatter_sub(delta, use_locking=self._use_locking)

    def _resource_apply_dense(self, grad, handle):
        return tf.train.GradientDescentOptimizer._resource_apply_dense(self=self, grad=grad, handle=handle)
        # return training_ops.resource_apply_gradient_descent(
        #     handle.handle, math_ops.cast(self._learning_rate_tensor,
        #                                  grad.dtype.base_dtype),
        #     grad, use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return tf.train.GradientDescentOptimizer._resource_apply_sparse_duplicate_indices(self=self, grad=grad, handle=handle)
        # return resource_variable_ops.resource_scatter_add(
        #     handle.handle, indices, -grad * self._learning_rate)
