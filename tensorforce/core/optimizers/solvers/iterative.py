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

import tensorflow as tf

from tensorforce.core.optimizers.solvers import Solver


class Iterative(Solver):

    def __init__(self, max_iterations):
        assert max_iterations >= 0
        self.max_iterations = max_iterations

        super(Iterative, self).__init__()

        self.initialize = tf.make_template(name_='solver-initialize', func_=self.tf_initialize)
        self.step = tf.make_template(name_='solver-step', func_=self.tf_step)
        self.next_step = tf.make_template(name_='solver-next_step', func_=self.tf_next_step)

    def tf_solve(self, fn_x, x_init, *args):
        self.fn_x = fn_x

        initial_args = self.initialize(x_init, *args)

        final_args = tf.while_loop(cond=self.next_step, body=self.step, loop_vars=initial_args)

        # Return first argument containing solution
        return final_args[0]

    def tf_initialize(self, x_init, *args):
        """
        First step of iterative solver.

        Args:
            x_init:
            *args:

        Returns: Initial value, iteration count

        """
        return x_init, 0

    def tf_step(self, x, iteration, *args):
        iteration += 1
        return (x, iteration) + args

    def tf_next_step(self, x, iteration, *args):
        return iteration < self.max_iterations
