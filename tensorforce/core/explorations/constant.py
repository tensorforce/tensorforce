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
from tensorforce.core.explorations import Exploration


class Constant(Exploration):
    """
    Explore via adding a constant term.
    """

    def __init__(self, constant=0.0, scope='constant', summary_labels=()):
        self.constant = constant
        super(Constant, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_explore(self, episode, timestep, shape):
        return tf.constant(value=self.constant, dtype=util.tf_dtype('float'), shape=shape)
