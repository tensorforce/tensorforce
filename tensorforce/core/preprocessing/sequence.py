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


class Sequence(Preprocessor):
    """
    Concatenate `length` state vectors. Example: Used in Atari
    problems to create the Markov property.
    """

    def __init__(self, length=2):
        super(Sequence, self).__init__()
        self.length = length

    def process(self, state):
        tf.assert_equal(x=tf.shape(input=state)[0], y=1)  # or just always the same?

        states_buffer = tf.get_variable(name='states-buffer', shape=((self.length,) + util.shape(state)[1:]), dtype=state.dtype, trainable=False)
        index = tf.get_variable(name='index', dtype=util.tf_dtype('int'), initializer=-1, trainable=False)

        assignment = tf.cond(
            pred=tf.equal(x=index, y=-1),
            true_fn=(lambda: tf.assign(
                ref=states_buffer,
                value=tf.tile(input=state, multiples=((self.length,) + tuple(1 for _ in range(util.rank(state) - 1))))
            )),
            false_fn=(lambda: tf.assign(ref=states_buffer[index], value=state[0]))
        )

        with tf.control_dependencies(control_inputs=(assignment,)):
            previous_states = [states_buffer[(index - n - 1) % self.length] for n in range(self.length)]
            assignment = tf.assign(ref=index, value=((tf.maximum(x=index, y=0) + 1) % self.length))

        with tf.control_dependencies(control_inputs=(assignment,)):
            return tf.expand_dims(input=tf.concat(values=previous_states, axis=-1), axis=0)

    def processed_shape(self, shape):
        return shape[:-1] + (shape[-1] * self.length,)

    def reset(self):
        # self.index = -1 !!!!!!!!!!!!
        pass
