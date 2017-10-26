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

    def __init__(self, states_spec, actions_spec, network_spec, config):
        # Likelihood ratio clipping
        assert config.likelihood_ratio_clipping is None or config.likelihood_ratio_clipping > 0.0
        self.likelihood_ratio_clipping = config.likelihood_ratio_clipping

        super(PGProbRatioModel, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=network_spec,
            config=config
        )

    def initialize(self, custom_getter):
        super(PGProbRatioModel, self).initialize(custom_getter)

        # Model comparison functions
        self.reference = tf.make_template(
            name_='reference',
            func_=self.tf_reference,
            custom_getter_=custom_getter
        )
        self.compare = tf.make_template(
            name_='compare',
            func_=self.tf_compare,
            custom_getter_=custom_getter
        )

    def tf_pg_loss_per_instance(self, states, internals, actions, terminal, reward):
        embedding = self.network.apply(x=states, internals=internals)
        prob_ratios = list()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parameters(x=embedding)
            log_prob = distribution.log_probability(distr_params=distr_params, action=actions[name])
            # works the same?
            # fixed_distr = tuple(tf.stop_gradient(input=x) for x in distr)
            # fixed_log_prob = distribution.log_probability(distr_params=fixed_distr, action=a)
            fixed_log_prob = tf.stop_gradient(input=log_prob)
            prob_ratio = tf.exp(x=(log_prob - fixed_log_prob))
            collapsed_size = util.prod(util.shape(prob_ratio)[1:])
            prob_ratio = tf.reshape(tensor=prob_ratio, shape=(-1, collapsed_size))
            prob_ratios.append(prob_ratio)
        prob_ratio = tf.reduce_mean(input_tensor=tf.concat(values=prob_ratios, axis=1), axis=1)
        if self.likelihood_ratio_clipping is None:
            return -prob_ratio * reward
        else:
            clipped_prob_ratio = tf.clip_by_value(
                t=prob_ratio,
                clip_value_min=(1.0 / (1.0 + self.likelihood_ratio_clipping)),
                clip_value_max=(1.0 + self.likelihood_ratio_clipping)
            )
            return -tf.minimum(x=(prob_ratio * reward), y=(clipped_prob_ratio * reward))

    def tf_reference(self, states, internals, actions):
        embedding = self.network.apply(x=states, internals=internals)
        log_probs = list()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parameters(x=embedding)
            log_prob = distribution.log_probability(distr_params=distr_params, action=actions[name])
            collapsed_size = util.prod(util.shape(log_prob)[1:])
            log_prob = tf.reshape(tensor=log_prob, shape=(-1, collapsed_size))
            log_probs.append(log_prob)
        return tf.reduce_mean(input_tensor=tf.concat(values=log_probs, axis=1), axis=1)

    def tf_compare(self, states, internals, actions, terminal, reward, reference):
        embedding = self.network.apply(x=states, internals=internals)
        log_probs = list()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parameters(x=embedding)
            log_prob = distribution.log_probability(distr_params=distr_params, action=actions[name])
            collapsed_size = util.prod(util.shape(log_prob)[1:])
            log_prob = tf.reshape(tensor=log_prob, shape=(-1, collapsed_size))
            log_probs.append(log_prob)
        log_prob = tf.reduce_mean(input_tensor=tf.concat(values=log_probs, axis=1), axis=1)
        prob_ratio = tf.exp(x=(log_prob - reference))
        return tf.reduce_mean(input_tensor=(prob_ratio * reward), axis=0)

    def get_optimizer_kwargs(self, states, actions, terminal, reward, internals):
        kwargs = super(PGProbRatioModel, self).get_optimizer_kwargs(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )
        kwargs['fn_reference'] = (
            lambda: self.reference(
                states=states,
                actions=actions,
                internals=internals
            )
        )
        kwargs['fn_compare'] = (
            lambda reference: self.compare(
                states=states,
                actions=actions,
                terminal=terminal,
                reward=reward,
                internals=internals,
                reference=reference
            )
        )
        return kwargs
