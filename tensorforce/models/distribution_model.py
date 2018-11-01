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
import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.core.networks import Network
from tensorforce.core.distributions import Distribution, Bernoulli, Categorical, Gaussian, Beta
from tensorforce.models import MemoryModel


class DistributionModel(MemoryModel):
    """
    Base class for models using distributions parametrized by a neural network.
    """

    COMPONENT_NETWORK = "network"
    COMPONENT_DISTRIBUTION = "distribution"

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
        requires_deterministic
    ):
        self.network_spec = network
        self.distributions_spec = distributions

        # Entropy regularization
        assert entropy_regularization is None or entropy_regularization >= 0.0
        self.entropy_regularization = entropy_regularization

        # For deterministic action sampling (Q vs PG model)
        self.requires_deterministic = requires_deterministic

        self.network = None
        self.distributions = None
        self.fn_kl_divergence = None

        super(DistributionModel, self).__init__(
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
            discount=discount
        )

    def setup_components_and_tf_funcs(self, custom_getter=None):
        """
        Creates and stores Network and Distribution objects.
        Generates and stores all template functions.
        """
        # Create network before super-call, since non-empty internals_spec attribute (for RNN) is required subsequently.
        self.network = Network.from_spec(
            spec=self.network_spec,
            kwargs=dict(summary_labels=self.summary_labels)
        )

        # Now that we have the network component: We can create the internals placeholders.
        assert len(self.internals_spec) == 0
        self.internals_spec = self.network.internals_spec()
        for name in sorted(self.internals_spec):
            internal = self.internals_spec[name]
            self.internals_input[name] = tf.placeholder(
                dtype=util.tf_dtype(internal['type']),
                shape=(None,) + tuple(internal['shape']),
                name=('internal-' + name)
            )
            if internal['initialization'] == 'zeros':
                self.internals_init[name] = np.zeros(shape=internal['shape'])
            else:
                raise TensorForceError("Invalid internal initialization value.")

        # And only then call super.
        custom_getter = super(DistributionModel, self).setup_components_and_tf_funcs(custom_getter)

        # Distributions
        self.distributions = self.create_distributions()

        # KL divergence function
        self.fn_kl_divergence = tf.make_template(
            name_='kl-divergence',
            func_=self.tf_kl_divergence,
            custom_getter_=custom_getter
        )

        return custom_getter

    def create_distributions(self):
        """
        Creates and returns the Distribution objects based on self.distributions_spec.

        Returns: Dict of distributions according to self.distributions_spec.
        """
        distributions = dict()
        for name in sorted(self.actions_spec):
            action = self.actions_spec[name]

            if self.distributions_spec is not None and name in self.distributions_spec:
                kwargs = dict(action)
                kwargs['scope'] = name
                kwargs['summary_labels'] = self.summary_labels
                distributions[name] = Distribution.from_spec(
                    spec=self.distributions_spec[name],
                    kwargs=kwargs
                )

            elif action['type'] == 'bool':
                distributions[name] = Bernoulli(
                    shape=action['shape'],
                    scope=name,
                    summary_labels=self.summary_labels
                )

            elif action['type'] == 'int':
                distributions[name] = Categorical(
                    shape=action['shape'],
                    num_actions=action['num_actions'],
                    scope=name,
                    summary_labels=self.summary_labels
                )

            elif action['type'] == 'float':
                if 'min_value' in action:
                    distributions[name] = Beta(
                        shape=action['shape'],
                        min_value=action['min_value'],
                        max_value=action['max_value'],
                        scope=name,
                        summary_labels=self.summary_labels
                    )

                else:
                    distributions[name] = Gaussian(
                        shape=action['shape'],
                        scope=name,
                        summary_labels=self.summary_labels
                    )

        return distributions

    def tf_actions_and_internals(self, states, internals, deterministic):
        embedding, internals = self.network.apply(
            x=states,
            internals=internals,
            update=tf.constant(value=False),
            return_internals=True
        )

        actions = dict()
        for name in sorted(self.distributions):
            distribution = self.distributions[name]
            distr_params = distribution.parameterize(x=embedding)
            actions[name] = distribution.sample(
                distr_params=distr_params,
                deterministic=tf.logical_or(x=deterministic, y=self.requires_deterministic)
            )
            # Prefix named variable with "name_" if more than 1 distribution.
            if len(self.distributions) > 1:
                name_prefix = name + "_"
            else:
                name_prefix = ""
            # parameterize() returns list as [logits, probabilities, state_value]
            self.network.set_named_tensor(name_prefix + "logits", distr_params[0])
            self.network.set_named_tensor(name_prefix + "probabilities", distr_params[1])
            self.network.set_named_tensor(name_prefix + "state_value", distr_params[2])

        return actions, internals

    def tf_regularization_losses(self, states, internals, update):
        losses = super(DistributionModel, self).tf_regularization_losses(
            states=states,
            internals=internals,
            update=update
        )

        network_loss = self.network.regularization_loss()
        if network_loss is not None:
            losses['network'] = network_loss

        for name in sorted(self.distributions):
            regularization_loss = self.distributions[name].regularization_loss()
            if regularization_loss is not None:
                if 'distributions' in losses:
                    losses['distributions'] += regularization_loss
                else:
                    losses['distributions'] = regularization_loss

        if (self.entropy_regularization is not None and self.entropy_regularization > 0.0) \
                or 'entropy' in self.summary_labels:
            entropies = list()
            embedding = self.network.apply(x=states, internals=internals, update=update)
            for name in sorted(self.distributions):
                distribution = self.distributions[name]
                distr_params = distribution.parameterize(x=embedding)
                entropy = distribution.entropy(distr_params=distr_params)
                collapsed_size = util.prod(util.shape(entropy)[1:])
                entropy = tf.reshape(tensor=entropy, shape=(-1, collapsed_size))
                entropies.append(entropy)
            entropy_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=entropies, axis=1), axis=1)
            entropy = tf.reduce_mean(input_tensor=entropy_per_instance, axis=0)

        if 'entropy' in self.summary_labels:
            tf.contrib.summary.scalar(name='entropy', tensor=entropy)
        if self.entropy_regularization is not None and self.entropy_regularization > 0.0:
            losses['entropy'] = -self.entropy_regularization * entropy

        return losses

    def tf_kl_divergence(self, states, internals, actions, terminal, reward, next_states, next_internals, update, reference=None):
        embedding = self.network.apply(x=states, internals=internals, update=update)
        kl_divergences = list()

        for name in sorted(self.distributions):
            distribution = self.distributions[name]
            distr_params = distribution.parameterize(x=embedding)
            fixed_distr_params = tuple(tf.stop_gradient(input=value) for value in distr_params)
            kl_divergence = distribution.kl_divergence(distr_params1=fixed_distr_params, distr_params2=distr_params)
            collapsed_size = util.prod(util.shape(kl_divergence)[1:])
            kl_divergence = tf.reshape(tensor=kl_divergence, shape=(-1, collapsed_size))
            kl_divergences.append(kl_divergence)

        kl_divergence_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=kl_divergences, axis=1), axis=1)
        return tf.reduce_mean(input_tensor=kl_divergence_per_instance, axis=0)

    def optimizer_arguments(self, states, internals, actions, terminal, reward, next_states, next_internals):
        arguments = super(DistributionModel, self).optimizer_arguments(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            next_states=next_states,
            next_internals=next_internals
        )
        arguments['fn_kl_divergence'] = self.fn_kl_divergence
        return arguments

    def get_variables(self, include_submodules=False, include_nontrainable=False):
        model_variables = super(DistributionModel, self).get_variables(
            include_submodules=include_submodules,
            include_nontrainable=include_nontrainable
        )

        network_variables = self.network.get_variables(include_nontrainable=include_nontrainable)
        model_variables += network_variables

        distribution_variables = [
            variable for name in sorted(self.distributions)
            for variable in self.distributions[name].get_variables(include_nontrainable=include_nontrainable)
        ]
        model_variables += distribution_variables

        return model_variables

    def get_components(self):
        result = dict(super(DistributionModel, self).get_components())
        result[DistributionModel.COMPONENT_NETWORK] = self.network
        for name in sorted(self.distributions):
            result["%s_%s" % (DistributionModel.COMPONENT_DISTRIBUTION, name)] = self.distributions[name]
        if len(self.distributions) == 1:
            result[DistributionModel.COMPONENT_DISTRIBUTION] = self.distributions[next(iter(sorted(self.distributions)))]
        return result
