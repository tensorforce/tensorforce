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
from tensorforce.core import distribution_modules, Module, network_modules, tf_function
from tensorforce.core.networks import Network
from tensorforce.core.policies import Stochastic, ActionValue


class ParametrizedDistributions(Stochastic, ActionValue):
    """
    Policy which parametrizes independent distributions per action conditioned on the output of a
    central states-processing neural network (supports both stochastic and action-value-based
    policy interface) (specification key: `parametrized_distributions`).

    Args:
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
        infer_states_value (bool): Experimental, whether to infer state value from distribution
            parameters (<span style="color:#00C000"><b>default</b></span>: false).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, network='auto', distributions=None, temperature=0.0, device=None, summary_labels=None,
        l2_regularization=None, name=None, states_spec=None, auxiliaries_spec=None,
        internals_spec=None, actions_spec=None
    ):
        super().__init__(
            temperature=temperature, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization, name=name, states_spec=states_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec
        )

        # Network
        self.network = self.add_module(
            name='network', module=network, modules=network_modules, inputs_spec=self.states_spec
        )
        output_spec = self.network.output_spec()
        if output_spec['type'] != 'float':
            raise TensorforceError(
                "Invalid output type for network: {}.".format(output_spec['type'])
            )
        embedding_shape = output_spec['shape']

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
                name=(name + '_distribution'), module=module, modules=distribution_modules,
                default_module=default_module, action_spec=spec, embedding_shape=embedding_shape
            )

        # States value
        if infer_states_value:
            self.value = None
        else:
            self.value = self.add_module(
                name='states-value', module='linear', modules=layer_modules, size=0,
                input_spec=output_spec
            )

    # (requires network as first argument)
    @classmethod
    def internals_spec(cls, network='auto', policy=None, **kwargs):
        internals_spec = super().internals_spec()

        if policy is None:
            assert 'name' in kwargs and 'states_spec' in kwargs
            network_cls, first_arg, network_kwargs = Module.get_module_class_and_kwargs(
                name='network', module=network, modules=network_modules,
                inputs_spec=kwargs['states_spec']
            )

            if first_arg is None:
                for name, spec in network_cls.internals_spec(**network_kwargs).items():
                    internals_spec['{}-{}'.format(kwargs['name'], name)] = spec
            else:
                for name, spec in network_cls.internals_spec(first_arg, **network_kwargs).items():
                    internals_spec['{}-{}'.format(kwargs['name'], name)] = spec

        else:
            assert network == 'auto' and len(kwargs) == 0
            for name, spec in policy.network.internals_spec(network=policy.network).items():
                internals_spec['{}-{}'.format(policy.name, name)] = spec

        return internals_spec

    def internals_init(self):
        internals_init = super().internals_init()

        for name, init in self.network.internals_init().items():
            internals_init['{}-{}'.format(self.name, name)] = init

        return internals_init

    def max_past_horizon(self, on_policy):
        return self.network.max_past_horizon(on_policy=on_policy)

    @tf_function(num_args=0)
    def past_horizon(self, on_policy):
        return self.network.past_horizon(on_policy=on_policy)

    @tf_function(num_args=4)
    def act(self, states, horizons, internals, auxiliaries, return_internals):
        return Stochastic.act(
            self=self, states=states, horizons=horizons, internals=internals,
            auxiliaries=auxiliaries, return_internals=return_internals
        )

    @tf_function(num_args=5)
    def sample_actions(
        self, states, horizons, internals, auxiliaries, temperatures, return_internals
    ):
        internals = util.fmap(
            function=(lambda x: x[len(self.name) + 1:]), xs=internals, depth=1, map_keys=True
        )

        if return_internals:
            embedding, internals = self.network.apply(
                x=states, horizons=horizons, internals=internals, return_internals=True
            )
            internals = util.fmap(
                function=(lambda x: self.name + '-' + x), xs=internals, depth=1, map_keys=True
            )
        else:
            embedding = self.network.apply(
                x=states, horizons=horizons, internals=internals, return_internals=False
            )

        actions = OrderedDict()
        for name, spec, distribution, temperature in util.zip_items(
            self.actions_spec, self.distributions, temperatures
        ):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            actions[name] = distribution.sample(parameters=parameters, temperature=temperature)

        if return_internals:
            return actions, internals
        else:
            return actions

    @tf_function(num_args=5)
    def log_probabilities(self, states, horizons, internals, auxiliaries, actions):
        internals = util.fmap(
            function=(lambda x: x[len(self.name) + 1:]), xs=internals, depth=1, map_keys=True
        )

        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, return_internals=False
        )

        log_probabilities = OrderedDict()
        for name, spec, distribution in util.zip_items(self.actions_spec, self.distributions):
            if spec['type'] == 'int':
                mask = auxiliaries[list(auxiliaries).index(name + '_mask')]
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            log_probabilities[name] = distribution.log_probability(
                parameters=parameters, action=actions[n]
            )

        return log_probabilities

    @tf_function(num_args=4)
    def entropies(self, states, horizons, internals, auxiliaries):
        internals = util.fmap(
            function=(lambda x: x[len(self.name) + 1:]), xs=internals, depth=1, map_keys=True
        )

        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, return_internals=False
        )

        entropies = OrderedDict()
        for name, distribution in util.zip_items(self.actions_spec, self.distributions):
            if spec['type'] == 'int':
                mask = auxiliaries[list(auxiliaries).index(name + '_mask')]
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            entropies[name] = distribution.entropy(parameters=parameters)

        return entropies

    @tf_function(num_args=5)
    def kl_divergences(self, states, horizons, internals, auxiliaries, other):
        parameters = self.kldiv_reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        )

        assert False
        if other is None:
            other = util.fmap(function=tf.stop_gradient, xs=parameters)
        elif isinstance(other, ParametrizedDistributions):
            other = other.kldiv_reference(
                states=states, internals=internals, auxiliaries=auxiliaries
            )
            other = util.fmap(function=tf.stop_gradient, xs=other)
        elif isinstance(other, dict):
            if any(name not in other for name in self.actions_spec):
                raise TensorforceError.unexpected()
        else:
            raise TensorforceError.unexpected()

        kl_divergences = OrderedDict()
        for name, distribution in self.distributions.items():
            kl_divergences[name] = distribution.kl_divergence(
                parameters1=parameters[n], parameters2=other[n]
            )

        return kl_divergences

    @tf_function(num_args=4)
    def kldiv_reference(self, states, horizons, internals, auxiliaries):
        internals = util.fmap(
            function=(lambda x: x[len(self.name) + 1:]), xs=internals, depth=1, map_keys=True
        )

        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, return_internals=False
        )

        kldiv_reference = OrderedDict()
        for name, spec, distribution in util.zip_items(self.actions_spec, self.distributions):
            if spec['type'] == 'int':
                mask = auxiliaries[list(auxiliaries).index(name + '_mask')]
                kldiv_reference[name] = distribution.parametrize(x=embedding, mask=mask)
            else:
                kldiv_reference[name] = distribution.parametrize(x=embedding)

        return kldiv_reference

    @tf_function(num_args=4)
    def states_values(self, states, horizons, internals, auxiliaries):
        internals = util.fmap(
            function=(lambda x: x[len(self.name) + 1:]), xs=internals, depth=1, map_keys=True
        )

        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, return_internals=False
        )

        states_values = OrderedDict()
        for name, spec, distribution in util.zip_items(self.actions_spec, self.distributions):
            if spec['type'] == 'int':
                mask = auxiliaries[name + '_mask']
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            states_values[name] = distribution.states_value(parameters=parameters)

        return states_values

    @tf_function(num_args=5)
    def actions_values(self, states, horizons, internals, auxiliaries, actions):
        internals = util.fmap(
            function=(lambda x: x[len(self.name) + 1:]), xs=internals, depth=1, map_keys=True
        )

        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, return_internals=False
        )

        actions_values = OrderedDict()
        for name, spec, distribution, action in util.zip_items(
            self.actions_spec, self.distributions, actions
        ):
            if spec['type'] == 'int':
                mask = auxiliaries[list(auxiliaries).index(name + '_mask')]
                parameters = distribution.parametrize(x=embedding, mask=mask)
            else:
                parameters = distribution.parametrize(x=embedding)
            actions_values[name] = distribution.action_value(parameters=parameters, action=action)

        return actions_values

    @tf_function(num_args=4)
    def states_value(
        self, states, internals, auxiliaries, reduced=True, include_per_action=False
    ):
        if self.value is None:
            return ActionValue.states_value(
                self=self, states=states, internals=internals, auxiliaries=auxiliaries,
                reduced=reduced, include_per_action=include_per_action
            )

        else:
            if not reduced or include_per_action:
                raise TensorforceError.invalid(name='policy.states_value', argument='reduced')

            embedding = self.network.apply(x=states, internals=internals)

            states_value = self.value.apply(x=embedding)

            return states_value
