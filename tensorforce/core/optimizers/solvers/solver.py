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
from tensorforce import util
import tensorforce.core.optimizers.solvers


class Solver(object):
    """
    Generic TensorFlow solver. Solves optimisation problems in pure TensorFlow.
    """

    def __init__(self):
        self.solve = tf.make_template(name_='solver', func_=self.tf_solve)

    def tf_solve(self, fn_x, *args):
        """

        Args:
            fn_x:
            *args:

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def from_config(config, kwargs=None):
        return util.get_object(
            obj=config,
            predefined=tensorforce.core.optimizers.solvers.solvers,
            kwargs=kwargs
        )
