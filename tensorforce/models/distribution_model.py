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
from tensorforce.core.networks import Network, LayeredNetwork
from tensorforce.core.distributions import Distribution, Bernoulli, Categorical, Gaussian, Beta
from tensorforce.models import Model


class DistributionModel(Model):
    """
    Base class for models using distributions
    """

    def __init__(self, states_spec, actions_spec, network_spec, config):
        with tf.name_scope(name=config.scope):
            # Network
            self.network = util.get_object(obj=network_spec, default_object=LayeredNetwork)
            assert isinstance(self.network, Network)

            # Distributions
            self.distributions = dict()
            for name, action in actions_spec.items():
                with tf.variable_scope(name_or_scope=(name + '-distribution')):
                    if config.distributions is not None and name in config.distributions:
                        self.distributions[name] = Distribution.from_spec(spec=config.distributions[name], kwargs=action)
                    elif action['type'] == 'bool':
                        self.distributions[name] = Bernoulli(shape=action['shape'])
                    elif action['type'] == 'int':
                        self.distributions[name] = Categorical(shape=action['shape'], num_actions=action['num_actions'])
                    elif action['type'] == 'float':
                        if 'min_value' in action:
                            self.distributions[name] = Beta(shape=action['shape'], min_value=action['min_value'], max_value=action['max_value'])
                        else:
                            self.distributions[name] = Gaussian(shape=action['shape'])

        # Entropy regularization
        assert config.entropy_regularization is None or config.entropy_regularization > 0.0
        self.entropy_regularization = config.entropy_regularization

        super(DistributionModel, self).__init__(states_spec, actions_spec, config)

    def initialize(self, custom_getter):
        super(DistributionModel, self).initialize(custom_getter)

        # Network internals
        self.internal_inputs.extend(self.network.internal_inputs())
        self.internal_inits.extend(self.network.internal_inits())

        # KL divergence function
        self.fn_kl_divergence = tf.make_template(
            name_='kl_divergence',
            func_=self.tf_kl_divergence,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )

    def tf_actions_and_internals(self, states, internals, deterministic):
        embedding, internals = self.network.apply(x=states, internals=internals, return_internals=True)
        actions = dict()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parameters(x=embedding)
            actions[name] = distribution.sample(distr_params=distr_params, deterministic=deterministic)
        return actions, internals

    def tf_regularization_losses(self, states, internals):
        losses = super(DistributionModel, self).tf_regularization_losses(states=states, internals=internals)

        network_loss = self.network.regularization_losses()
        if network_loss is not None:
            losses['network'] = network_loss

        if self.entropy_regularization is not None:
            entropies = list()
            embedding = self.network.apply(x=states, internals=internals)
            for name, distribution in self.distributions.items():
                distr_params = distribution.parameters(x=embedding)
                entropy = distribution.entropy(distr_params=distr_params)
                collapsed_size = util.prod(util.shape(entropy)[1:])
                entropy = tf.reshape(tensor=entropy, shape=(-1, collapsed_size))
                entropies.append(entropy)

            entropy_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=entropies, axis=1), axis=1)
            entropy = tf.reduce_mean(input_tensor=entropy_per_instance, axis=0)
            losses['entropy'] = -self.entropy_regularization * entropy

        return losses

    def tf_kl_divergence(self, states, internals):
        embedding = self.network.apply(x=states, internals=internals)
        kl_divergences = list()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parameters(x=embedding)
            fixed_distr_params = tuple(tf.stop_gradient(input=value) for value in distr_params)
            kl_divergence = distribution.kl_divergence(distr_params1=fixed_distr_params, distr_params2=distr_params)
            collapsed_size = util.prod(util.shape(kl_divergence)[1:])
            kl_divergence = tf.reshape(tensor=kl_divergence, shape=(-1, collapsed_size))
            kl_divergences.append(kl_divergence)
        kl_divergence_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=kl_divergences, axis=1), axis=1)
        return tf.reduce_mean(input_tensor=kl_divergence_per_instance, axis=0)

    def get_optimizer_kwargs(self, states, actions, terminal, reward, internals):
        kwargs = super(DistributionModel, self).get_optimizer_kwargs(states, actions, terminal, reward, internals)
        kwargs['fn_kl_divergence'] = (lambda: self.fn_kl_divergence(states=states, internals=internals))
        return kwargs

    def get_variables(self):
        return super(DistributionModel, self).get_variables() + self.network.get_variables() + [variable for name in sorted(self.distributions) for variable in self.distributions[name].get_variables()]
