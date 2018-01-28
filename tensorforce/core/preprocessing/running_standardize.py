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
from tensorforce.core.preprocessing import Preprocessor


class RunningStandardize(Preprocessor):
    """
    Standardize state w.r.t past states.
    Subtract mean and divide by standard deviation of sequence of past states.
    """

    def __init__(
        self,
        shape,
        axis=None,
        reset_after_batch=True,
        scope='running_standardize',
        summary_labels=()
    ):
        self.axis = axis
        self.reset_after_batch = reset_after_batch
        super(RunningStandardize, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def reset(self):
        if self.reset_after_batch:
            # TODO is this being called as a tf op?
            pass

    def tf_process(self, tensor):
        state = tensor
        count = tf.get_variable(
            name='count',
            dtype=tf.int32,
            initializer=0,
            trainable=False
        )
        mean_estimate = tf.get_variable(
            name='mean-estimate',
            shape=self.shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        variance_sum_estimate = tf.get_variable(
            name='variance-sum-estimate',
            shape=self.shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        with tf.control_dependencies([tf.assign_add(ref=count, value=1)]):
            # TODO Do we want to allow an axis or always standardize along time
            update_mean = tf.reduce_sum(input_tensor=(state - mean_estimate), axis=self.axis)

            # Update implements: https://www.johndcook.com/blog/standard_deviation/
            # TODO check batch shapes?
            float_count = tf.cast(x=count, dtype=tf.float32)
            mean_estimate = tf.cond(
                pred=(count > 1),
                true_fn=(lambda: mean_estimate + update_mean / float_count),
                false_fn=(lambda: mean_estimate)
            )
            update_variance_sum = (state - mean_estimate) * (state - update_mean)
            variance_sum_estimate = variance_sum_estimate + \
                tf.reduce_sum(input_tensor=update_variance_sum, axis=self.axis)

            variance_estimate = tf.cond(
                pred=(count > 1),
                true_fn=(lambda: variance_sum_estimate / (float_count - 1.0)),
                false_fn=(lambda: variance_estimate)
            )
        # print('mean estimate shape = {}'.format(tf.shape(mean_estimate)))

        return tf.cond(
            pred=(count > 1),
            true_fn=(lambda: (state - mean_estimate) / (tf.maximum(x=tf.sqrt(x=variance_estimate), y=util.epsilon))),
            false_fn=(lambda: state)
        )
