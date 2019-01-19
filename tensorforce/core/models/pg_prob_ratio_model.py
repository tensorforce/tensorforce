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


class PGProbRatioModel(PGModel):
    """
    Policy gradient model based on computing likelihood ratios, e.g. TRPO and PPO.
    """

    def __init__(
        self,
        # Model
        states, actions, scope, device, saver, summarizer, execution, parallel_interactions,
        buffer_observe, exploration, variable_noise, states_preprocessing, reward_preprocessing,
        # MemoryModel
        update_mode, memory, optimizer, discount,
        # DistributionModel
        network, distributions, entropy_regularization,
        # PGModel
        baseline_mode, baseline, baseline_optimizer, gae_lambda,
        # PGProbRatioModel
        likelihood_ratio_clipping
    ):
        super().__init__(
            # Model
            states=states, actions=actions, scope=scope, device=device, saver=saver,
            summarizer=summarizer, execution=execution,
            parallel_interactions=parallel_interactions, buffer_observe=buffer_observe,
            exploration=exploration, variable_noise=variable_noise,
            states_preprocessing=states_preprocessing, reward_preprocessing=reward_preprocessing,
            # MemoryModel
            update_mode=update_mode, memory=memory, optimizer=optimizer, discount=discount,
            # DistributionModel
            network=network, distributions=distributions,
            entropy_regularization=entropy_regularization,
            # PGModel
            baseline_mode=baseline_mode, baseline=baseline, baseline_optimizer=baseline_optimizer,
            gae_lambda=gae_lambda
        )

        # Likelihood ratio clipping (10.0 essentially like no clipping)
        likelihood_ratio_clipping = 10.0 if likelihood_ratio_clipping is None \
            else likelihood_ratio_clipping
        self.likelihood_ratio_clipping = self.add_module(
            name='likelihood-ratio-clipping', module=likelihood_ratio_clipping,
            modules=parameter_modules, dtype='float'
        )

    def tf_reference(
        self, states, internals, actions, terminal, reward, next_states, next_internals
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
        return tf.stop_gradient(input=log_probs)

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
        if reference is None:
            old_log_probs = tf.stop_gradient(input=log_probs)
        else:
            old_log_probs = reference

        # Comment on log_ratio 1.0 and gradient perspective
        prob_ratios = tf.exp(x=(log_probs - old_log_probs))
        prob_ratio_per_instance = tf.reduce_mean(input_tensor=prob_ratios, axis=1)

        likelihood_ratio_clipping = self.likelihood_ratio_clipping.value()

        clipped_prob_ratio_per_instance = tf.clip_by_value(
            t=prob_ratio_per_instance,
            clip_value_min=(1.0 / (1.0 + likelihood_ratio_clipping)),
            clip_value_max=(1.0 + likelihood_ratio_clipping)
        )
        return -tf.minimum(
            x=(prob_ratio_per_instance * reward),
            y=(clipped_prob_ratio_per_instance * reward)
        )

        # def no_clipping():
        #     return -prob_ratio_per_instance * reward

        # def apply_clipping():
        #     clipped_prob_ratio_per_instance = tf.clip_by_value(
        #         t=prob_ratio_per_instance,
        #         clip_value_min=(1.0 / (1.0 + likelihood_ratio_clipping)),
        #         clip_value_max=(1.0 + likelihood_ratio_clipping)
        #     )
        #     return -tf.minimum(
        #         x=(prob_ratio_per_instance * reward),
        #         y=(clipped_prob_ratio_per_instance * reward)
        #     )

        # zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        # skip_clipping = tf.math.equal(x=likelihood_ratio_clipping, y=zero)
        # return self.cond(pred=skip_clipping, true_fn=no_clipping, false_fn=apply_clipping)
