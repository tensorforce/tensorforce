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

from collections import OrderedDict

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import distribution_modules, Module, network_modules
from tensorforce.core.models import MemoryModel


class DistributionModel(MemoryModel):
    """
    Base class for models using distributions parametrized by a neural network.
    """

    def __init__(
        self,
        # Model
        states, actions, scope, device, saver, summarizer, execution, parallel_interactions,
        buffer_observe, variable_noise, states_preprocessing, actions_exploration,
        reward_preprocessing,
        # MemoryModel
        update_mode, memory, optimizer, discount,
        # DistributionModel
        network, distributions, entropy_regularization, requires_deterministic
    ):
        super().__init__(
            # Model
            states=states, actions=actions, scope=scope, device=device, saver=saver,
            summarizer=summarizer, execution=execution,
            parallel_interactions=parallel_interactions, buffer_observe=buffer_observe,
            variable_noise=variable_noise, states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration, reward_preprocessing=reward_preprocessing,
            # MemoryModel
            update_mode=update_mode, memory=memory, optimizer=optimizer, discount=discount
        )

        # Network
        self.network = self.add_module(
            name='network', module=network, modules=network_modules, default_module='layered',
            inputs_spec=self.states_spec
        )
        output_spec = self.network.get_output_spec()
        if output_spec['type'] != 'float':
            raise TensorforceError(
                "Invalid output type for network: {}.".format(output_spec['type'])
            )
        elif len(output_spec['shape']) != 1:
            raise TensorforceError(
                "Invalid output rank for network: {}.".format(len(output_spec['shape']))
            )
        embedding_size = output_spec['shape'][0]

        # Distributions
        self.distributions = OrderedDict()
        for name, action_spec in self.actions_spec.items():
            if action_spec['type'] == 'bool':
                default_module = 'bernoulli'
            elif action_spec['type'] == 'int':
                default_module = 'categorical'
            elif action_spec['type'] == 'float':
                if 'min_value' in action_spec:
                    default_module = 'beta'
                else:
                    default_module = 'gaussian'

            if distributions is None:
                module = None
            else:
                module = dict()
                if action_spec['type'] in distributions:
                    module.update(distributions[name])
                if name in distributions:
                    module.update(distributions[name])

            self.distributions[name] = self.add_module(
                name=(name + '-distribution'), module=module, modules=distribution_modules,
                default_module=default_module, action_spec=action_spec,
                embedding_size=embedding_size
            )

        # Entropy regularization
        assert entropy_regularization is None or entropy_regularization >= 0.0
        self.entropy_regularization = entropy_regularization

        # For deterministic action sampling (Q vs PG model)
        self.requires_deterministic = requires_deterministic

        # Internals specification
        self.internals_spec.update(self.network.internals_spec())
        for name in self.internals_spec:
            if name in self.states_spec:
                raise TensorforceError(
                    "Name overlap between internals and states: {}.".format(name)
                )
            if name in self.actions_spec:
                raise TensorforceError(
                    "Name overlap between internals and actions: {}.".format(name)
                )
        self.internals_init.update(self.network.internals_init())

    def tf_core_act(self, states, internals):
        super().tf_core_act(states=states, internals=internals)

        embedding, internals = self.network.apply(
            x=states, internals=internals, return_internals=True
        )

        actions = OrderedDict()
        for name, distribution in self.distributions.items():
            distr_params = distribution.parameterize(x=embedding)
            deterministic = Module.retrieve_tensor(name='deterministic')
            deterministic = tf.logical_or(
                x=deterministic,
                y=tf.constant(value=self.requires_deterministic, dtype=util.tf_dtype(dtype='bool'))
            )
            actions[name] = distribution.sample(
                distr_params=distr_params, deterministic=deterministic
            )

        return actions, internals

    def tf_regularize(self, states, internals):
        regularization_loss = super().tf_regularize(states=states, internals=internals)

        entropies = list()
        embedding = self.network.apply(x=states, internals=internals)
        for name in sorted(self.distributions):
            distribution = self.distributions[name]
            distr_params = distribution.parameterize(x=embedding)
            entropy = distribution.entropy(distr_params=distr_params)
            collapsed_size = util.product(xs=util.shape(entropy)[1:])
            entropy = tf.reshape(tensor=entropy, shape=(-1, collapsed_size))
            entropies.append(entropy)

        entropies = tf.concat(values=entropies, axis=1)
        entropy_per_instance = tf.reduce_mean(input_tensor=entropies, axis=1)
        entropy = tf.reduce_mean(input_tensor=entropy_per_instance, axis=0)

        self.add_summary(label='entropy', name='entropy', tensor=entropy)

        if self.entropy_regularization is not None and self.entropy_regularization > 0.0:
            regularization_loss = regularization_loss - self.entropy_regularization * entropy

        return regularization_loss

    def tf_kl_divergence(
        self, states, internals, actions, terminal, reward, next_states, next_internals,
        reference=None
    ):
        embedding = self.network.apply(x=states, internals=internals)

        kl_divergences = list()
        for name in sorted(self.distributions):
            distribution = self.distributions[name]
            distr_params = distribution.parameterize(x=embedding)
            fixed_distr_params = tuple(tf.stop_gradient(input=value) for value in distr_params)
            kl_divergence = distribution.kl_divergence(
                distr_params1=fixed_distr_params, distr_params2=distr_params
            )
            collapsed_size = util.product(xs=util.shape(kl_divergence)[1:])
            kl_divergence = tf.reshape(tensor=kl_divergence, shape=(-1, collapsed_size))
            kl_divergences.append(kl_divergence)

        kl_divergences = tf.concat(values=kl_divergences, axis=1)
        kl_divergence_per_instance = tf.reduce_mean(input_tensor=kl_divergences, axis=1)
        return tf.reduce_mean(input_tensor=kl_divergence_per_instance, axis=0)

    def optimizer_arguments(self, states, internals, actions, terminal, reward, next_states, next_internals):
        arguments = super().optimizer_arguments(
            states=states, internals=internals, actions=actions, terminal=terminal, reward=reward,
            next_states=next_states, next_internals=next_internals
        )

        arguments['fn_kl_divergence'] = self.kl_divergence
        return arguments
