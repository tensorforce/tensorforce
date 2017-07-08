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


class EvolutionaryOptimizer(tf.train.Optimizer):

    def __init__(self, learning_rate, use_locking=False, name="Evolutionary"):
        # """Construct a new gradient descent optimizer.
        # Args:
        #   learning_rate: A Tensor or a floating point value.  The learning
        #     rate to use.
        #   use_locking: If True use locks for update operations.
        #   name: Optional name prefix for the operations created when applying
        #     gradients. Defaults to "GradientDescent".
        # """
        super(EvolutionaryOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self.stddev = 0.00001

    def compute_gradients(self, loss, var_list=None, gate_gradients=None, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None):
        if gate_gradients is None:
            gate_gradients = EvolutionaryOptimizer.GATE_OP
        if gate_gradients not in [EvolutionaryOptimizer.GATE_NONE, EvolutionaryOptimizer.GATE_OP, EvolutionaryOptimizer.GATE_GRAPH]:
            raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" % gate_gradients)
        self._assert_valid_dtypes([loss])
        if grad_loss is not None:
            self._assert_valid_dtypes([grad_loss])
        if var_list is None:
            var_list = (tf.trainable_variables() + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        else:
            var_list = tf.python.util.nest.flatten(var_list)
        # pylint: disable=protected-access
        var_list += tf.get_collection(tf.GraphKeys._STREAMING_MODEL_PORTS)
        # pylint: enable=protected-access
        # processors = [_get_processor(v) for v in var_list]
        if not var_list:
            raise ValueError("No variables to optimize.")
        # var_refs = [p.target() for p in processors]
        # grads = gradients.gradients(loss, var_refs, grad_ys=grad_loss, gate_gradients=(gate_gradients == Optimizer.GATE_OP), aggregation_method=aggregation_method, colocate_gradients_with_ops=colocate_gradients_with_ops)
        perturbations = [tf.random_normal(shape=var.get_shape()) * self.stddev for var in var_list]
        unperturbed_loss = tf.identity(input=loss)
        # unperturbed_loss = tf.Print(unperturbed_loss, (unperturbed_loss, var_list[0]))
        with tf.control_dependencies(control_inputs=(unperturbed_loss,)):
            ops = list()
            for n, var in enumerate(var_list):
                ops.append(var.assign_add(delta=perturbations[n]))
            with tf.control_dependencies(control_inputs=ops):
                # degradation = loss - unperturbed_loss
                # degradation = tf.Print(degradation, (degradation, loss, unperturbed_loss, var_list[0]))
                grads = [tf.multiply(x=loss, y=perturbation) / self.stddev for perturbation in perturbations]
                with tf.control_dependencies(control_inputs=grads):
                    ops = list()
                    for n, var in enumerate(var_list):
                        ops.append(var.assign_sub(delta=perturbations[n]))
                    with tf.control_dependencies(control_inputs=ops):
                        if gate_gradients == EvolutionaryOptimizer.GATE_GRAPH:
                            grads = tf.python.ops.control_flow_ops.tuple(grads)
                        grads_and_vars = list(zip(grads, var_list))
                        self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None and v.dtype != tf.resource])
        return grads_and_vars

    def _apply_dense(self, grad, var):
        return tf.train.GradientDescentOptimizer._apply_dense(self=self, grad=grad, var=var)

    def _resource_apply_dense(self, grad, handle):
        return tf.train.GradientDescentOptimizer._resource_apply_dense(self=self, grad=grad, handle=handle)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return tf.train.GradientDescentOptimizer._resource_apply_sparse_duplicate_indices(self=self, grad=grad, handle=handle, indices=indices)

    def _apply_sparse_duplicate_indices(self, grad, var):
        return tf.train.GradientDescentOptimizer._apply_sparse_duplicate_indices(self=self, grad=grad, var=var)

    def _prepare(self):
        return tf.train.GradientDescentOptimizer._prepare(self=self)
