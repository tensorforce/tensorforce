# Copyright 2018 Tensorforce Team. All Rights Reserved.
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
from tensorforce.core import parameter_modules
from tensorforce.core.models import PGModel


class PGLogProbModel(PGModel):
    """
    Policy gradient model based on computing log likelihoods, e.g. the classical REINFORCE
    algorithm.
    """

    def tf_loss_per_instance(
        self, states, internals, actions, terminal, reward, next_states, next_internals,
        reference=None
    ):
        embedding = self.network.apply(x=states, internals=internals)

        log_probs = list()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parametrize(x=embedding)
            action = actions[name]
            log_prob = distribution.log_probability(distr_params=distr_params, action=action)
            collapsed_size = util.product(xs=util.shape(log_prob)[1:])
            log_prob = tf.reshape(tensor=log_prob, shape=(-1, collapsed_size))
            log_probs.append(log_prob)

        log_probs = tf.concat(values=log_probs, axis=1)
        log_prob_per_instance = tf.reduce_mean(input_tensor=log_probs, axis=1)
        return -log_prob_per_instance * reward
