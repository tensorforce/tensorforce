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


class EpsilonAnneal(Exploration):
    """
    Annealing epsilon parameter based on ratio of current timestep to total timesteps.
    """

    def __init__(
        self,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        timesteps=10000,
        start_timestep=0,
        scope='epsilon_anneal',
        summary_labels=()
    ):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.timesteps = timesteps
        self.start_timestep = start_timestep

        super(EpsilonAnneal, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_explore(self, episode, timestep, shape):

        def true_fn():
            # Know if first is not true second must be true from outer cond check.
            return tf.cond(
                pred=(timestep < self.start_timestep),
                true_fn=(lambda: self.initial_epsilon),
                false_fn=(lambda: self.final_epsilon)
            )

        def false_fn():
            completed_ratio = (tf.cast(x=timestep, dtype=util.tf_dtype('float')) - self.start_timestep) / self.timesteps
            return self.initial_epsilon + completed_ratio * (self.final_epsilon - self.initial_epsilon)

        pred = tf.logical_or(x=(timestep < self.start_timestep), y=(timestep > self.start_timestep + self.timesteps))
        return tf.fill(dims=shape, value=tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn))
