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

from tensorforce import util
from tensorforce.models import PGModel


class PGProbRatioModel(PGModel):
    """
    Policy gradient model based on computing likelihood ratios, e.g. TRPO and PPO.
    """

    def __init__(
        self,
        states,
        actions,
        scope,
        device,
        saver,
        summarizer,
        execution,
        batching_capacity,
        variable_noise,
        states_preprocessing,
        actions_exploration,
        reward_preprocessing,
        update_mode,
        memory,
        optimizer,
        discount,
        network,
        distributions,
        entropy_regularization,
        baseline_mode,
        baseline,
        baseline_optimizer,
        gae_lambda,
        likelihood_ratio_clipping
    ):
        # Likelihood ratio clipping
        assert likelihood_ratio_clipping is None or likelihood_ratio_clipping > 0.0
        self.likelihood_ratio_clipping = likelihood_ratio_clipping

        # self.reference = None
        # self.compare = None

        super(PGProbRatioModel, self).__init__(
            states=states,
            actions=actions,
            scope=scope,
            device=device,
            saver=saver,
            summarizer=summarizer,
            execution=execution,
            batching_capacity=batching_capacity,
            variable_noise=variable_noise,
            states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration,
            reward_preprocessing=reward_preprocessing,
            update_mode=update_mode,
            memory=memory,
            optimizer=optimizer,
            discount=discount,
            network=network,
            distributions=distributions,
            entropy_regularization=entropy_regularization,
            baseline_mode=baseline_mode,
            baseline=baseline,
            baseline_optimizer=baseline_optimizer,
            gae_lambda=gae_lambda
        )

    def tf_reference(self, states, internals, actions, terminal, reward, next_states, next_internals, update):
        embedding = self.network.apply(x=states, internals=internals, update=update)

        log_probs = list()
        for name in sorted(self.distributions):
            distribution = self.distributions[name]
            distr_params = distribution.parameterize(x=embedding)
            log_prob = distribution.log_probability(distr_params=distr_params, action=actions[name])
            collapsed_size = util.prod(util.shape(log_prob)[1:])
            log_prob = tf.reshape(tensor=log_prob, shape=(-1, collapsed_size))
            log_probs.append(log_prob)

        return tf.stop_gradient(input=tf.concat(values=log_probs, axis=1))

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward, next_states, next_internals, update, reference=None):
        embedding = self.network.apply(x=states, internals=internals, update=update)

        log_probs = list()
        for name in sorted(self.distributions):
            distribution = self.distributions[name]
            distr_params = distribution.parameterize(x=embedding)
            log_prob = distribution.log_probability(distr_params=distr_params, action=actions[name])
            collapsed_size = util.prod(util.shape(log_prob)[1:])
            log_prob = tf.reshape(tensor=log_prob, shape=(-1, collapsed_size))
            log_probs.append(log_prob)

        log_prob = tf.concat(values=log_probs, axis=1)
        if reference is None:
            old_log_prob = tf.stop_gradient(input=log_prob)
        else:
            old_log_prob = reference

        prob_ratio = tf.exp(x=(log_prob - old_log_prob))
        prob_ratio = tf.reduce_mean(input_tensor=prob_ratio, axis=1)

        if self.likelihood_ratio_clipping is None:
            return -prob_ratio * reward

        else:
            clipped_prob_ratio = tf.clip_by_value(
                t=prob_ratio,
                clip_value_min=(1.0 / (1.0 + self.likelihood_ratio_clipping)),
                clip_value_max=(1.0 + self.likelihood_ratio_clipping)
            )
            return -tf.minimum(x=(prob_ratio * reward), y=(clipped_prob_ratio * reward))
