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
from tensorforce.core.networks import Network
from tensorforce.core.policies import Stochastic, ActionValue


class ParametrizedDistributions(Stochastic, ActionValue):
    """
    Policy which parametrizes independent distributions per action conditioned on the output of a
    central states-processing neural network (supports both stochastic and action-value-based
    policy interface) (specification key: `parametrized_distributions`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        states_spec (specification): States specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        actions_spec (specification): Actions specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        network ('auto' | specification): Policy network configuration, see
            [networks](../modules/networks.html)
            (<span style="color:#00C000"><b>default</b></span>: 'auto', automatically configured
            network).
        distributions (dict[specification]): Distributions configuration, see
            [distributions](../modules/distributions.html), specified per
            action-type or -name
            (<span style="color:#00C000"><b>default</b></span>: per action-type, Bernoulli
            distribution for binary boolean actions, categorical distribution for discrete integer
            actions, Gaussian distribution for unbounded continuous actions, Beta distribution for
            bounded continuous actions).
        temperature (parameter | dict[parameter], float >= 0.0): Sampling temperature, global or
            per action (<span style="color:#00C000"><b>default</b></span>: 0.0).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, states_spec, actions_spec, network='auto', distributions=None, temperature=0.0,
        device=None, summary_labels=None, l2_regularization=None
    ):
        if isinstance(network, Network):
            assert device is None
            device = network.device
            network.device = None

        super().__init__(
            name=name, states_spec=states_spec, actions_spec=actions_spec, temperature=temperature,
            device=device, summary_labels=summary_labels, l2_regularization=l2_regularization
        )

        # Network
        self.network = self.add_module(
            name=(self.name + '-network'), module=network, modules=network_modules,
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
        Module.register_tensor(name=self.name, spec=output_spec, batched=True)
        embedding_size = output_spec['shape'][0]

        # Distributions
        self.distributions = OrderedDict()
        for name, spec in self.actions_spec.items():
            if spec['type'] == 'bool':
                default_module = 'bernoulli'
            elif spec['type'] == 'int':
                default_module = 'categorical'
            elif spec['type'] == 'float':
                default_module = 'beta' if 'min_value' in spec else 'gaussian'

            if distributions is None:
                module = None
            else:
                module = dict()
                if spec['type'] in distributions:
                    if isinstance(distributions[spec['type']], str):
                        module = distributions[spec['type']]
                    else:
                        module.update(distributions[spec['type']])
                if name in distributions:
                    if isinstance(distributions[name], str):
                        module = distributions[name]
                    else:
                        module.update(distributions[name])

            self.distributions[name] = self.add_module(
                name=(name + '-distribution'), module=module, modules=distribution_modules,
                default_module=default_module, action_spec=spec, embedding_size=embedding_size
            )

    @classmethod
    def internals_spec(cls, network=None, policy=None, name=None, states_spec=None, **kwargs):
        if policy is None:
            if network is None:
                network = 'auto'
            assert name is not None and states_spec is not None

            network_cls, first_arg, kwargs = Module.get_module_class_and_kwargs(
                name=(name + '-network'), module=network, modules=network_modules,
                inputs_spec=states_spec
            )

            if first_arg is None:
                return network_cls.internals_spec(name=(name + '-network'), **kwargs)
            else:
                return network_cls.internals_spec(first_arg, name=(name + '-network'), **kwargs)

        else:
            assert network is None and name is None and states_spec is None
            return policy.network.__class__.internals_spec(network=policy.network)

    def internals_init(self):
        return self.network.internals_init()

    def tf_dependency_horizon(self, is_optimization=False):
        return self.network.dependency_horizon(is_optimization=is_optimization)

    def tf_act(self, states, internals, auxiliaries, return_internals):
        return Stochastic.tf_act(
            self=self, states=states, internals=internals, auxiliaries=auxiliaries,
            return_internals=return_internals
        )

    def tf_sample_actions(self, states, internals, auxiliaries, temperature, return_internals):
        if return_internals:
            embedding, internals = self.network.apply(
                x=states, internals=internals, return_internals=return_internals
            )
        else:
            embedding = self.network.apply(
                x=states, internals=internals, return_internals=return_internals
            )

        Module.update_tensor(name=self.name, tensor=embedding)

        actions = OrderedDict()
        for name, spec, distribution, temp in util.zip_items(
            self.actions_spec, self.distributions, temperature
        ):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            action = distribution.sample(parameters=parameters, temperature=temp)

            entropy = distribution.entropy(parameters=parameters)
            entropy = tf.reshape(tensor=entropy, shape=(-1, util.product(xs=spec['shape'])))
            mean_entropy = tf.reduce_mean(input_tensor=entropy, axis=1)
            actions[name] = self.add_summary(
                label='entropy', name=(name + '-entropy'), tensor=mean_entropy, pass_tensors=action
            )

        if return_internals:
            return actions, internals
        else:
            return actions

    def tf_log_probabilities(self, states, internals, auxiliaries, actions):
        embedding = self.network.apply(x=states, internals=internals)
        Module.update_tensor(name=self.name, tensor=embedding)

        log_probabilities = OrderedDict()
        for name, spec, distribution, action in util.zip_items(
            self.actions_spec, self.distributions, actions
        ):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            log_probabilities[name] = distribution.log_probability(
                parameters=parameters, action=action
            )

        return log_probabilities

    def tf_entropies(self, states, internals, auxiliaries):
        embedding = self.network.apply(x=states, internals=internals)
        Module.update_tensor(name=self.name, tensor=embedding)

        entropies = OrderedDict()
        for name, spec, distribution in util.zip_items(self.actions_spec, self.distributions):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            entropies[name] = distribution.entropy(parameters=parameters)

        return entropies

    def tf_kl_divergences(self, states, internals, auxiliaries, other=None):
        assert other is None or isinstance(other, ParametrizedDistributions)

        embedding = self.network.apply(x=states, internals=internals)
        if other is not None:
            other_embedding = other.network.apply(x=states, internals=internals)

        kl_divergences = OrderedDict()
        for name, spec, distribution in util.zip_items(self.actions_spec, self.distributions):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            if other is None:
                other_parameters = tuple(tf.stop_gradient(input=value) for value in parameters)
            elif spec['type'] == 'int':
                other_parameters = other.distributions[name].parametrize(
                    x=other_embedding, mask=mask
                )
            else:
                other_parameters = other.distributions[name].parametrize(x=other_embedding)
            kl_divergences[name] = distribution.kl_divergence(
                parameters1=other_parameters, parameters2=parameters  # order????
            )

        return kl_divergences

    def tf_states_values(self, states, internals, auxiliaries):
        embedding = self.network.apply(x=states, internals=internals)
        Module.update_tensor(name=self.name, tensor=embedding)

        states_values = OrderedDict()
        for name, spec, distribution in util.zip_items(self.actions_spec, self.distributions):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            states_values[name] = distribution.states_value(parameters=parameters)

        return states_values

    def tf_actions_values(self, states, internals, auxiliaries, actions=None):
        embedding = self.network.apply(x=states, internals=internals)
        Module.update_tensor(name=self.name, tensor=embedding)

        actions_values = OrderedDict()
        for name, spec, distribution, action in util.zip_items(
            self.actions_spec, self.distributions, actions
        ):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            actions_values[name] = distribution.action_value(parameters=parameters, action=action)

        return actions_values
