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
from tensorforce.core.networks import Network
from tensorforce.core.distributions import Distribution, Bernoulli, Categorical, Gaussian, Beta
from tensorforce.models import Model


class DistributionModel(Model):
    """
    Base class for models using distributions parameterized by a neural network
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        network_spec,
        device,
        scope,
        saver_spec,
        summary_spec,
        distributed_spec,
        optimizer,
        discount,
        normalize_rewards,
        variable_noise,
        distributions_spec,
        entropy_regularization
    ):
        self.network_spec = network_spec
        self.distributions_spec = distributions_spec

        # Entropy regularization
        assert entropy_regularization is None or entropy_regularization >= 0.0
        self.entropy_regularization = entropy_regularization

        super(DistributionModel, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=network_spec,
            device=device,
            scope=scope,
            saver_spec=saver_spec,
            summary_spec=summary_spec,
            distributed_spec=distributed_spec,
            optimizer=optimizer,
            discount=discount,
            normalize_rewards=normalize_rewards,
            variable_noise=variable_noise
        )

    def initialize(self, custom_getter):
        super(DistributionModel, self).initialize(custom_getter)

        # Network
        self.network = Network.from_spec(
            spec=self.network_spec,
            kwargs=dict(summary_labels=self.summary_labels)
        )

        # Distributions
        self.distributions = self.generate_distributions(self.actions_spec, self.distributions_spec, self.summary_labels)

        # Network internals
        self.internal_inputs.extend(self.network.internal_inputs())
        self.internal_inits.extend(self.network.internal_inits())

        # KL divergence function
        self.fn_kl_divergence = tf.make_template(
            name_=(self.scope + '/kl-divergence'),
            func_=self.tf_kl_divergence,
            custom_getter_=custom_getter
        )

    @staticmethod
    def generate_distributions(actions_spec, distributions_spec, summary_labels):
        distributions = dict()
        for name, action in actions_spec.items():
            with tf.name_scope(name=(name + '-distribution')):

                if distributions_spec is not None and name in distributions_spec:
                    kwargs = dict(action)
                    kwargs['summary_labels'] = summary_labels
                    distributions[name] = Distribution.from_spec(
                        spec=distributions_spec[name],
                        kwargs=kwargs
                    )

                elif action['type'] == 'bool':
                    distributions[name] = Bernoulli(
                        shape=action['shape'],
                        summary_labels=summary_labels
                    )

                elif action['type'] == 'int':
                    distributions[name] = Categorical(
                        shape=action['shape'],
                        num_actions=action['num_actions'],
                        summary_labels=summary_labels
                    )

                elif action['type'] == 'float':
                    if 'min_value' in action:
                        distributions[name] = Beta(
                            shape=action['shape'],
                            min_value=action['min_value'],
                            max_value=action['max_value'],
                            summary_labels=summary_labels
                        )

                    else:
                        distributions[name] = Gaussian(
                            shape=action['shape'],
                            summary_labels=summary_labels
                        )

        return distributions

    def tf_actions_and_internals(self, states, internals, update, deterministic):
        embedding, internals = self.network.apply(x=states, internals=internals, update=update, return_internals=True)
        actions = dict()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parameterize(x=embedding)
            actions[name] = distribution.sample(distr_params=distr_params, deterministic=deterministic)
        return actions, internals

    def tf_kl_divergence(self, states, internals, update):
        embedding = self.network.apply(x=states, internals=internals, update=update)
        kl_divergences = list()

        for name, distribution in self.distributions.items():
            distr_params = distribution.parameterize(x=embedding)
            fixed_distr_params = tuple(tf.stop_gradient(input=value) for value in distr_params)
            kl_divergence = distribution.kl_divergence(distr_params1=fixed_distr_params, distr_params2=distr_params)
            collapsed_size = util.prod(util.shape(kl_divergence)[1:])
            kl_divergence = tf.reshape(tensor=kl_divergence, shape=(-1, collapsed_size))
            kl_divergences.append(kl_divergence)

        kl_divergence_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=kl_divergences, axis=1), axis=1)
        return tf.reduce_mean(input_tensor=kl_divergence_per_instance, axis=0)

    def get_optimizer_kwargs(self, states, internals, actions, terminal, reward, update):
        kwargs = super(DistributionModel, self).get_optimizer_kwargs(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            update=update
        )
        kwargs['fn_kl_divergence'] = (
            lambda: self.fn_kl_divergence(
                states=states,
                internals=internals,
                update=update
            )
        )
        return kwargs

    def tf_regularization_losses(self, states, internals, update):
        losses = super(DistributionModel, self).tf_regularization_losses(
            states=states,
            internals=internals,
            update=update
        )

        network_loss = self.network.regularization_loss()
        if network_loss is not None:
            losses['network'] = network_loss

        for distribution in self.distributions.values():
            regularization_loss = distribution.regularization_loss()
            if regularization_loss is not None:
                if 'distributions' in losses:
                    losses['distributions'] += regularization_loss
                else:
                    losses['distributions'] = regularization_loss

        if self.entropy_regularization is not None and self.entropy_regularization > 0.0:
            entropies = list()
            embedding = self.network.apply(x=states, internals=internals, update=update)
            for name, distribution in self.distributions.items():
                distr_params = distribution.parameterize(x=embedding)
                entropy = distribution.entropy(distr_params=distr_params)
                collapsed_size = util.prod(util.shape(entropy)[1:])
                entropy = tf.reshape(tensor=entropy, shape=(-1, collapsed_size))
                entropies.append(entropy)

            entropy_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=entropies, axis=1), axis=1)
            entropy = tf.reduce_mean(input_tensor=entropy_per_instance, axis=0)
            losses['entropy'] = -self.entropy_regularization * entropy

        return losses

    def get_variables(self, include_non_trainable=False):
        model_variables = super(DistributionModel, self).get_variables(include_non_trainable=include_non_trainable)
        network_variables = self.network.get_variables(include_non_trainable=include_non_trainable)
        distribution_variables = self.get_distributions_variables(self.distributions, include_non_trainable=include_non_trainable)

        return model_variables + network_variables + distribution_variables

    def get_summaries(self):
        model_summaries = super(DistributionModel, self).get_summaries()
        network_summaries = self.network.get_summaries()
        distribution_summaries = self.get_distributions_summaries(self.distributions)

        return model_summaries + network_summaries + distribution_summaries

    @staticmethod
    def get_distributions_variables(distributions, include_non_trainable=False):
        distribution_variables = [
            variable for name in sorted(distributions)
            for variable in distributions[name].get_variables(include_non_trainable=include_non_trainable)
        ]
        return distribution_variables

    @staticmethod
    def get_distributions_summaries(distributions):
        distribution_summaries = [
            summary for name in sorted(distributions)
            for summary in distributions[name].get_summaries()
        ]
        return distribution_summaries
