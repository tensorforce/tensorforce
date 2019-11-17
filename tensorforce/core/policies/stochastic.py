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

from tensorforce import util
from tensorforce.core import Module, parameter_modules
from tensorforce.core.policies import Policy


class Stochastic(Policy):
    """
    Base class for stochastic policies.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        states_spec (specification): States specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        actions_spec (specification): Actions specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
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
        self, name, states_spec, actions_spec, temperature=0.0, device=None, summary_labels=None,
        l2_regularization=None
    ):
        super().__init__(
            name=name, states_spec=states_spec, actions_spec=actions_spec, device=device,
            summary_labels=summary_labels, l2_regularization=l2_regularization
        )

        # Sampling temperature
        if isinstance(temperature, dict) and \
                all(name in self.actions_spec for name in temperature):
            # Different temperature per action
            self.temperature = OrderedDict()
            for name in self.actions_spec:
                if name in temperature:
                    self.temperature[name] = self.add_module(
                        name=(name + '-temperature'), module=temperature[name],
                        modules=parameter_modules, is_trainable=False, dtype='float'
                    )
        else:
            # Same temperature for all actions
            self.temperature = self.add_module(
                name='temperature', module=temperature, modules=parameter_modules,
                is_trainable=False, dtype='float'
            )

    def tf_act(self, states, internals, auxiliaries, return_internals):
        deterministic = Module.retrieve_tensor(name='deterministic')

        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        temperature = OrderedDict()
        if isinstance(self.temperature, dict):
            for name in self.actions_spec:
                if name in self.temperature:
                    temperature[name] = tf.where(
                        condition=deterministic, x=zero, y=self.temperature[name].value()
                    )
                else:
                    temperature[name] = zero
        else:
            value = tf.where(condition=deterministic, x=zero, y=self.temperature.value())
            for name in self.actions_spec:
                temperature[name] = value

        return self.sample_actions(
            states=states, internals=internals, auxiliaries=auxiliaries,
            temperature=temperature, return_internals=return_internals
        )

    def tf_log_probability(
        self, states, internals, auxiliaries, actions, reduced=True, include_per_action=False
    ):
        log_probabilities = self.log_probabilities(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions
        )

        for name, spec, log_probability in util.zip_items(self.actions_spec, log_probabilities):
            log_probabilities[name] = tf.reshape(
                tensor=log_probability, shape=(-1, util.product(xs=spec['shape']))
            )

        log_probability = tf.concat(values=tuple(log_probabilities.values()), axis=1)
        if reduced:
            log_probability = tf.math.reduce_mean(input_tensor=log_probability, axis=1)
            if include_per_action:
                for name in self.actions_spec:
                    log_probabilities[name] = tf.math.reduce_mean(
                        input_tensor=log_probabilities[name], axis=1
                    )

        if include_per_action:
            log_probabilities['*'] = log_probability
            return log_probabilities
        else:
            return log_probability

    def tf_entropy(self, states, internals, auxiliaries, reduced=True, include_per_action=False):
        entropies = self.entropies(states=states, internals=internals, auxiliaries=auxiliaries)

        for name, spec, entropy in util.zip_items(self.actions_spec, entropies):
            entropies[name] = tf.reshape(
                tensor=entropy, shape=(-1, util.product(xs=spec['shape']))
            )

        entropy = tf.concat(values=tuple(entropies.values()), axis=1)

        if reduced:
            entropy = tf.math.reduce_mean(input_tensor=entropy, axis=1)
            if include_per_action:
                for name in self.actions_spec:
                    entropies[name] = tf.math.reduce_mean(input_tensor=entropies[name], axis=1)

        if include_per_action:
            entropies['*'] = entropy
            return entropies
        else:
            return entropy

    def tf_kl_divergence(
        self, states, internals, auxiliaries, other=None, reduced=True, include_per_action=False
    ):
        kl_divergences = self.kl_divergences(
            states=states, internals=internals, auxiliaries=auxiliaries, other=other
        )

        for name, spec, kl_divergence in util.zip_items(self.actions_spec, kl_divergences):
            kl_divergences[name] = tf.reshape(
                tensor=kl_divergence, shape=(-1, util.product(xs=spec['shape']))
            )

        kl_divergence = tf.concat(values=tuple(kl_divergences.values()), axis=1)
        if reduced:
            kl_divergence = tf.math.reduce_mean(input_tensor=kl_divergence, axis=1)
            if include_per_action:
                for name in self.actions_spec:
                    kl_divergences[name] = tf.math.reduce_mean(
                        input_tensor=kl_divergences[name], axis=1
                    )

        if include_per_action:
            kl_divergences['*'] = kl_divergence
            return kl_divergences
        else:
            return kl_divergence

    def tf_sample_actions(self, states, internals, auxiliaries, temperature, return_internals):
        raise NotImplementedError

    def tf_log_probabilities(self, states, internals, auxiliaries, actions):
        raise NotImplementedError

    def tf_entropies(self, states, internals, auxiliaries):
        raise NotImplementedError

    def tf_kl_divergences(self, states, internals, auxiliaries, other=None):
        raise NotImplementedError

    def tf_kldiv_reference(self, states, internals, auxiliaries):
        raise NotImplementedError
