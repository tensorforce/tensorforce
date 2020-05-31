# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce import TensorforceError
from tensorforce.core import distribution_modules, layer_modules, ModuleDict, network_modules, \
    TensorDict, tf_function
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
        use_beta_distribution (bool): Whether to use the Beta distribution for bounded continuous
            actions by default.
            (<span style="color:#00C000"><b>default</b></span>: true).
        state_value_mode ("independent" | "max-action-values" | "infer-from-distributions" | "no-state-value" | "no-distributions"):
            How to estimate state value, either via a separate linear layer independent of
            distributions ("independent"), or inferred as maximum of discrete bool/int action
            values, or inferred from distribution parameters ("infer-from-distributions",
            experimental), or special degenerate "no-state-value"/"no-distributions" policy
            (<span style="color:#00C000"><b>default</b></span>: "independent").
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    # Network first
    def __init__(
        self, network='auto', *, distributions=None, temperature=0.0, use_beta_distribution=True,
        state_value_mode='independent', device=None,  l2_regularization=None, name=None,
        states_spec=None, auxiliaries_spec=None, internals_spec=None, actions_spec=None
    ):
        super().__init__(
            temperature=temperature, device=device, l2_regularization=l2_regularization, name=name,
            states_spec=states_spec, auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec
        )

        # Network
        self.network = self.submodule(
            name='network', module=network, modules=network_modules, inputs_spec=self.states_spec
        )
        output_spec = self.network.output_spec()
        if output_spec.type != 'float':
            raise TensorforceError.type(
                name='ParametrizedDistributions', argument='network output', dtype=output_spec.type
            )

        # State value mode
        if state_value_mode not in (
            'independent', 'max-action-values', 'infer-from-distributions', 'no-state-value',
            'no-distributions'
        ):
            raise TensorforceError.value(
                name='ParametrizedDistributions', argument='state_value_mode',
                value=state_value_mode, hint='not from {independent,max-action-values,infer-from-di'
                                             'stributions,no-state-value,no-distributions}'
            )
        self.state_value_mode = state_value_mode

        # Distributions
        if self.state_value_mode != 'no-distributions':
            self.distributions = ModuleDict()
            for name, spec in self.actions_spec.items():
                if spec.type == 'bool':
                    default_module = 'bernoulli'
                elif spec.type == 'int':
                    assert spec.num_values is not None
                    default_module = 'categorical'
                elif spec.type == 'float':
                    if use_beta_distribution and spec.min_value is not None:
                        default_module = 'beta'
                    else:
                        default_module = 'gaussian'

                if distributions is None:
                    module = None
                else:
                    module = dict()
                    if spec.type in distributions:
                        if isinstance(distributions[spec.type], str):
                            module = distributions[spec.type]
                        else:
                            module.update(distributions[spec.type])
                    if name in distributions:
                        if isinstance(distributions[name], str):
                            module = distributions[name]
                        else:
                            module.update(distributions[name])

                self.distributions[name] = self.submodule(
                    name=(name + '_distribution'), module=module, modules=distribution_modules,
                    default_module=default_module, action_spec=spec, input_spec=output_spec
                )

        # State value
        if self.state_value_mode == 'independent' or self.state_value_mode == 'no-distributions':
            self.value = self.submodule(
                name='states_value', module='linear', modules=layer_modules, size=0,
                input_spec=output_spec
            )

    @property
    def internals_spec(self):
        return self.network.internals_spec

    def internals_init(self):
        return self.network.internals_init()

    def max_past_horizon(self, *, on_policy):
        return self.network.max_past_horizon(on_policy=on_policy)

    @tf_function(num_args=0)
    def past_horizon(self, *, on_policy):
        return self.network.past_horizon(on_policy=on_policy)

    @tf_function(num_args=4)
    def act(self, *, states, horizons, internals, auxiliaries, independent, return_internals):
        if independent:
            embedding = self.network.apply(
                x=states, horizons=horizons, internals=internals, independent=independent,
                return_internals=return_internals
            )
            if return_internals:
                embedding, internals = embedding

            def function(name, distribution):
                conditions = auxiliaries.get(name, default=TensorDict())
                parameters = distribution.parametrize(x=embedding, conditions=conditions)
                return distribution.mode(parameters=parameters)

            actions = self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

            if return_internals:
                return actions, internals
            else:
                return actions

        else:
            return Stochastic.act(
                self=self, states=states, horizons=horizons, internals=internals,
                auxiliaries=auxiliaries, independent=independent, return_internals=return_internals
            )

    @tf_function(num_args=5)
    def sample_actions(
        self, *, states, horizons, internals, auxiliaries, temperatures, independent,
        return_internals
    ):
        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, independent=independent,
            return_internals=return_internals
        )
        if return_internals:
            embedding, internals = embedding

        def function(name, distribution, temperature):
            conditions = auxiliaries.get(name, default=TensorDict())
            parameters = distribution.parametrize(x=embedding, conditions=conditions)
            return distribution.sample(parameters=parameters, temperature=temperature)

        actions = self.distributions.fmap(
            function=function, cls=TensorDict, with_names=True, zip_values=temperatures
        )

        if return_internals:
            return actions, internals
        else:
            return actions

    @tf_function(num_args=5)
    def log_probabilities(self, *, states, horizons, internals, auxiliaries, actions):
        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, independent=True,
            return_internals=False
        )

        def function(name, distribution, action):
            conditions = auxiliaries.get(name, default=TensorDict())
            parameters = distribution.parametrize(x=embedding, conditions=conditions)
            return distribution.log_probability(parameters=parameters, action=action)

        return self.distributions.fmap(
            function=function, cls=TensorDict, with_names=True, zip_values=actions
        )

    @tf_function(num_args=4)
    def entropies(self, *, states, horizons, internals, auxiliaries):
        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, independent=True,
            return_internals=False
        )

        def function(name, distribution):
            conditions = auxiliaries.get(name, default=TensorDict())
            parameters = distribution.parametrize(x=embedding, conditions=conditions)
            return distribution.entropy(parameters=parameters)

        return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

    @tf_function(num_args=5)
    def kl_divergences(self, *, states, horizons, internals, auxiliaries, reference):
        parameters = self.kldiv_reference(
            states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        )
        reference = reference.fmap(function=tf.stop_gradient)

        def function(distribution, parameters1, parameters2):
            return distribution.kl_divergence(parameters1=parameters1, parameters2=parameters2)

        return self.distributions.fmap(
            function=function, cls=TensorDict, zip_values=(parameters, reference)
        )

    @tf_function(num_args=4)
    def kldiv_reference(self, *, states, horizons, internals, auxiliaries):
        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, independent=True,
            return_internals=False
        )

        def function(name, distribution):
            conditions = auxiliaries.get(name, default=TensorDict())
            return distribution.parametrize(x=embedding, conditions=conditions)

        return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

    @tf_function(num_args=4)
    def states_value(self, *, states, horizons, internals, auxiliaries, reduced, return_per_action):
        if self.state_value_mode == 'independent' or self.state_value_mode == 'no-distributions':
            if not reduced or return_per_action:
                raise TensorforceError.invalid(name='policy.states_value', argument='reduced')

            embedding = self.network.apply(
                x=states, horizons=horizons, internals=internals, independent=True,
            return_internals=False
            )

            return self.value.apply(x=embedding)

        else:
            return super().states_value(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                reduced=reduced, return_per_action=return_per_action
            )

    @tf_function(num_args=4)
    def states_values(self, *, states, horizons, internals, auxiliaries):
        if self.state_value_mode == 'max-action-values':
            return super().states_values(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
            )

        else:
            embedding = self.network.apply(
                x=states, horizons=horizons, internals=internals, independent=True,
                return_internals=False
            )

            def function(name, distribution):
                conditions = auxiliaries.get(name, default=TensorDict())
                parameters = distribution.parametrize(x=embedding, conditions=conditions)
                return distribution.states_value(parameters=parameters)

        return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

    @tf_function(num_args=5)
    def actions_values(self, *, states, horizons, internals, auxiliaries, actions):
        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, independent=True,
            return_internals=False
        )

        def function(name, distribution, action):
            conditions = auxiliaries.get(name, default=TensorDict())
            parameters = distribution.parametrize(x=embedding, conditions=conditions)
            return distribution.action_value(parameters=parameters, action=action)

        return self.distributions.fmap(
            function=function, cls=TensorDict, with_names=True, zip_values=actions
        )

    @tf_function(num_args=4)
    def all_actions_values(self, *, states, horizons, internals, auxiliaries):
        if not all(spec.type in ('bool', 'int') for spec in self.actions_spec.values()):
            raise TensorforceError.value(
                name='ParametrizedDistributions', argument='state_value_mode',
                value='max-action-values', condition='action types not bool/int'
            )

        embedding = self.network.apply(
            x=states, horizons=horizons, internals=internals, independent=True,
            return_internals=False
        )

        def function(name, distribution):
            conditions = auxiliaries.get(name, default=TensorDict())
            parameters = distribution.parametrize(x=embedding, conditions=conditions)
            return distribution.all_action_values(parameters=parameters)

        return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)
