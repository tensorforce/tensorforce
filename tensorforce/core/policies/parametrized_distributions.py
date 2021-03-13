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

from tensorforce.core import distribution_modules, ModuleDict, TensorDict, TensorsSpec, \
    tf_function, tf_util
from tensorforce.core.policies import ParametrizedPolicy, StochasticPolicy, ValuePolicy


class ParametrizedDistributions(StochasticPolicy, ValuePolicy, ParametrizedPolicy):
    """
    Policy which parametrizes independent distributions per action, conditioned on the output of a
    central neural network processing the input state, supporting both a stochastic and value-based
    policy interface (specification key: `parametrized_distributions`).

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
            per action (<span style="color:#00C000"><b>default</b></span>: 1.0).
        use_beta_distribution (bool): Whether to use the Beta distribution for bounded continuous
            actions by default.
            (<span style="color:#00C000"><b>default</b></span>: false).
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
        self, network='auto', *, distributions=None, temperature=1.0, use_beta_distribution=False,
        device=None, l2_regularization=None, name=None, states_spec=None, auxiliaries_spec=None,
        internals_spec=None, actions_spec=None
    ):
        super().__init__(
            temperature=temperature, device=device, l2_regularization=l2_regularization, name=name,
            states_spec=states_spec, auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec
        )

        ParametrizedPolicy.__init__(self=self, network=network, inputs_spec=self.states_spec)
        output_spec = self.network.output_spec()

        # Distributions
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
                if name is None and isinstance(distributions, str):
                    module = distributions
                elif name in distributions:
                    if isinstance(distributions[name], str):
                        module = distributions[name]
                    else:
                        module.update(distributions[name])
                elif spec.type in distributions:
                    if isinstance(distributions[spec.type], str):
                        module = distributions[spec.type]
                    else:
                        module.update(distributions[spec.type])
                elif name is None and 'type' in distributions:
                    module.update(distributions)

            if name is None:
                self.distributions[name] = self.submodule(
                    name='action_distribution', module=module, modules=distribution_modules,
                    default_module=default_module, action_spec=spec, input_spec=output_spec
                )
            else:
                self.distributions[name] = self.submodule(
                    name=(name + '_distribution'), module=module, modules=distribution_modules,
                    default_module=default_module, action_spec=spec, input_spec=output_spec
                )

        self.kldiv_reference_spec = self.distributions.fmap(
            function=(lambda x: x.parameters_spec), cls=TensorsSpec
        )

    def input_signature(self, *, function):
        try:
            return StochasticPolicy.input_signature(self=self, function=function)
        except NotImplementedError:
            return ValuePolicy.input_signature(self=self, function=function)

    def output_signature(self, *, function):
        try:
            return StochasticPolicy.output_signature(self=self, function=function)
        except NotImplementedError:
            return ValuePolicy.output_signature(self=self, function=function)

    def get_savedmodel_trackables(self):
        trackables = super().get_savedmodel_trackables()
        for distribution in self.distributions.values():
            for variable in distribution.variables:
                assert variable.name not in trackables
                trackables[variable.name] = variable
        return trackables

    @tf_function(num_args=5)
    def act(self, *, states, horizons, internals, auxiliaries, deterministic, independent):
        assertions = list()
        if self.config.create_tf_assertions:
            if not independent:
                false = tf_util.constant(value=False, dtype='bool')
                assertions.append(tf.debugging.assert_equal(x=deterministic, y=false))

        embedding, internals = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=independent
        )

        def fn_mode():

            def function(name, distribution):
                conditions = auxiliaries.get(name, default=TensorDict())
                parameters = distribution.parametrize(x=embedding, conditions=conditions)
                return distribution.mode(parameters=parameters, independent=independent)

            return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

        def fn_sample():
            if isinstance(self.temperature, dict):

                def function(name, distribution, temp):
                    conditions = auxiliaries.get(name, default=TensorDict())
                    parameters = distribution.parametrize(x=embedding, conditions=conditions)
                    return distribution.sample(
                        parameters=parameters, temperature=temp, independent=independent
                    )

                temperature = self.temperature.fmap(function=(lambda x: x.value()), cls=TensorDict)
                return self.distributions.fmap(
                    function=function, cls=TensorDict, with_names=True, zip_values=(temperature,)
                )

            else:
                temperature = self.temperature.value()

                def function(name, distribution):
                    conditions = auxiliaries.get(name, default=TensorDict())
                    parameters = distribution.parametrize(x=embedding, conditions=conditions)
                    return distribution.sample(
                        parameters=parameters, temperature=temperature, independent=independent
                    )

                return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

        with tf.control_dependencies(control_inputs=assertions):
            actions = tf.cond(pred=deterministic, true_fn=fn_mode, false_fn=fn_sample)
            return actions, internals

    @tf_function(num_args=5)
    def act_entropy(self, *, states, horizons, internals, auxiliaries, deterministic, independent):
        assertions = list()
        if self.config.create_tf_assertions:
            if not independent:
                false = tf_util.constant(value=False, dtype='bool')
                assertions.append(tf.debugging.assert_equal(x=deterministic, y=false))

        embedding, internals = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=independent
        )

        def fn_mode():

            def function(name, distribution):
                conditions = auxiliaries.get(name, default=TensorDict())
                parameters = distribution.parametrize(x=embedding, conditions=conditions)
                mode = distribution.mode(parameters=parameters, independent=independent)
                entropy = distribution.entropy(parameters=parameters)
                return mode, entropy

            return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

        def fn_sample():
            if isinstance(self.temperature, dict):

                def function(name, distribution, temp):
                    conditions = auxiliaries.get(name, default=TensorDict())
                    parameters = distribution.parametrize(x=embedding, conditions=conditions)
                    sample = distribution.sample(
                        parameters=parameters, temperature=temp, independent=independent
                    )
                    entropy = distribution.entropy(parameters=parameters)
                    return sample, entropy

                temperature = self.temperature.fmap(function=(lambda x: x.value()), cls=TensorDict)
                return self.distributions.fmap(
                    function=function, cls=TensorDict, with_names=True, zip_values=(temperature,)
                )

            else:
                temperature = self.temperature.value()

                def function(name, distribution):
                    conditions = auxiliaries.get(name, default=TensorDict())
                    parameters = distribution.parametrize(x=embedding, conditions=conditions)
                    sample = distribution.sample(
                        parameters=parameters, temperature=temperature, independent=independent
                    )
                    entropy = distribution.entropy(parameters=parameters)
                    return sample, entropy

                return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

        with tf.control_dependencies(control_inputs=assertions):
            actions, entropies = tf.cond(pred=deterministic, true_fn=fn_mode, false_fn=fn_sample)

            def function(value, spec):
                return tf.reshape(tensor=value, shape=(-1, spec.size))

            # See also implementation of StochasticPolicy.entropy()
            entropies = entropies.fmap(function=function, zip_values=self.actions_spec)
            entropies = tf.concat(values=tuple(entropies.values()), axis=1)
            entropy = tf.math.reduce_mean(input_tensor=entropies, axis=1)

            return actions, internals, entropy

    @tf_function(num_args=5)
    def log_probabilities(self, *, states, horizons, internals, auxiliaries, actions):
        deterministic = tf_util.constant(value=True, dtype='bool')
        embedding, _ = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=True
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
        deterministic = tf_util.constant(value=True, dtype='bool')
        embedding, _ = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=True
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
        deterministic = tf_util.constant(value=True, dtype='bool')
        embedding, _ = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=True
        )

        def function(name, distribution):
            conditions = auxiliaries.get(name, default=TensorDict())
            return distribution.parametrize(x=embedding, conditions=conditions)

        return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)

    @tf_function(num_args=5)
    def action_values(self, *, states, horizons, internals, auxiliaries, actions):
        deterministic = tf_util.constant(value=True, dtype='bool')
        embedding, _ = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=True
        )

        def function(name, distribution, action):
            conditions = auxiliaries.get(name, default=TensorDict())
            parameters = distribution.parametrize(x=embedding, conditions=conditions)
            return distribution.action_value(parameters=parameters, action=action)

        return self.distributions.fmap(
            function=function, cls=TensorDict, with_names=True, zip_values=actions
        )

    @tf_function(num_args=4)
    def state_values(self, *, states, horizons, internals, auxiliaries):
        deterministic = tf_util.constant(value=True, dtype='bool')
        embedding, _ = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=True
        )

        def function(name, distribution):
            conditions = auxiliaries.get(name, default=TensorDict())
            parameters = distribution.parametrize(x=embedding, conditions=conditions)
            return distribution.state_value(parameters=parameters)

        return self.distributions.fmap(function=function, cls=TensorDict, with_names=True)
