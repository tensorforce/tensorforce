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
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorforce import util
from tensorforce.core.preprocessors import Preprocessor


class RunningStandardize(Preprocessor):
    """
    Standardize state w.r.t past states.
    Subtract mean and divide by standard deviation of sequence of past states.
    Based on https://www.johndcook.com/blog/standard_deviation/.
    """

    def __init__(
        self,
        shape,
        reset_after_batch=True,
        scope='running_standardize',
        summary_labels=()
    ):
        self.reset_after_batch = reset_after_batch
        # The op that resets our stats variables.
        self.reset_op = None
        super(RunningStandardize, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_reset(self):
        if self.reset_after_batch:
            return [self.reset_op]

    def tf_process(self, tensor):
        count = tf.get_variable(
            name='count',
            dtype=util.tf_dtype('float'),
            initializer=0.0,
            trainable=False
        )
        mean_estimate = tf.get_variable(
            name='mean-estimate',
            shape=self.shape,
            dtype=util.tf_dtype('float'),
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        variance_sum_estimate = tf.get_variable(
            name='variance-sum-estimate',
            shape=self.shape,
            dtype=util.tf_dtype('float'),
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        self.reset_op = tf.variables_initializer([count, mean_estimate, variance_sum_estimate], name='reset-op')

        assignment = tf.assign_add(ref=count, value=1.0)

        with tf.control_dependencies(control_inputs=(assignment,)):
            # Mean update
            mean = tf.reduce_sum(input_tensor=(tensor - mean_estimate), axis=0)  # reduce_mean?
            assignment = tf.assign_add(ref=mean_estimate, value=(mean / count))

        with tf.control_dependencies(control_inputs=(assignment,)):

            def first_run():
                # No meaningful mean and variance yet.
                return tensor

            def later_run():
                # Variance update
                variance = tf.reduce_sum(input_tensor=((tensor - mean_estimate) * mean), axis=0)  # reduce_mean?
                assignment = tf.assign_add(ref=variance_sum_estimate, value=variance)
                with tf.control_dependencies(control_inputs=(assignment,)):
                    variance_estimate = variance_sum_estimate / (count - 1.0)
                    # Standardize tensor
                    return (tensor - mean_estimate) / tf.maximum(x=tf.sqrt(x=variance_estimate), y=util.epsilon)

            return tf.cond(pred=(count > 1.0), true_fn=later_run, false_fn=first_run)
