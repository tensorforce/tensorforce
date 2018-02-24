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


class Sequence(Preprocessor):
    """
    Concatenate `length` state vectors. Example: Used in Atari
    problems to create the Markov property (velocity of game objects as they move across the screen).
    """

    def __init__(self, shape, length=2, add_rank=False, scope='sequence', summary_labels=()):
        """
        Args:
            length (int): The number of states to concatenate. In the beginning, when no previous state is available,
                concatenate the given first state with itself `length` times.
            add_rank (bool): Whether to add another rank to the end of the input with dim=length-of-the-sequence.
                This could be useful if e.g. a grayscale image of w x h pixels is coming from the env
                (no color channel). The output of the preprocessor would then be of shape [batch] x w x h x [length].
        """
        # raise TensorForceError("The sequence preprocessor is temporarily broken; use version 0.3.2 if required.")
        self.length = length
        self.add_rank = add_rank
        # The op that resets index back to -1.
        self.reset_op = None
        super(Sequence, self).__init__(shape=shape, scope=scope, summary_labels=summary_labels)

    def tf_reset(self):
        return [self.reset_op]

    def tf_process(self, tensor):
        # or just always the same?
        tf.assert_equal(x=tf.shape(input=tensor)[0], y=1)

        states_buffer = tf.get_variable(
            name='states-buffer',
            shape=((self.length,) + util.shape(tensor)[1:]),
            dtype=tensor.dtype,
            trainable=False
        )
        index = tf.get_variable(
            name='index',
            dtype=util.tf_dtype('int'),
            initializer=-1,
            trainable=False
        )
        self.reset_op = tf.variables_initializer([index], name='reset-op')

        def first_run():
            fill_buffer = (self.length,) + tuple(1 for _ in range(util.rank(tensor) - 1))
            return tf.assign(ref=states_buffer, value=tf.tile(input=tensor, multiples=fill_buffer))

        def later_run():
            return tf.assign(ref=states_buffer[index], value=tensor[0])

        assignment = tf.cond(pred=(index >= 0), true_fn=later_run, false_fn=first_run)

        with tf.control_dependencies(control_inputs=(assignment,)):
            previous_states = [states_buffer[(index - n - 1) % self.length] for n in range(self.length)]
            assignment = tf.assign(ref=index, value=((tf.maximum(x=index, y=0) + 1) % self.length))

        with tf.control_dependencies(control_inputs=(assignment,)):
            if self.add_rank:
                stack = tf.stack(values=previous_states, axis=-1)
            else:
                stack = tf.concat(values=previous_states, axis=-1)
            batch_one = tf.expand_dims(input=stack, axis=0)
            return batch_one

    def processed_shape(self, shape):
        if self.add_rank:
            return shape + (self.length,)
        else:
            return shape[:-1] + (shape[-1] * self.length,)
