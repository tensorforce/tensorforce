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

import tensorflow as tf

from tensorforce import util
from tensorforce.core import Module
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
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def tf_act(self, states, internals, auxiliaries):
        deterministic = Module.retrieve_tensor(name='deterministic')

        return self.sample_actions(
            states=states, internals=internals, auxiliaries=auxiliaries,
            deterministic=deterministic, return_internals=True
        )

    def tf_log_probability(self, states, internals, auxiliaries, actions, mean=True):
        log_probabilities = self.log_probabilities(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions
        )

        for name, spec, log_probability in util.zip_items(self.actions_spec, log_probabilities):
            log_probabilities[name] = tf.reshape(
                tensor=log_probability, shape=(-1, util.product(xs=spec['shape']))
            )

        log_probability = tf.concat(values=tuple(log_probabilities.values()), axis=1)
        if mean:
            log_probability = tf.math.reduce_mean(input_tensor=log_probability, axis=1)

        return log_probability

    def tf_entropy(self, states, internals, auxiliaries, mean=True):
        entropies = self.entropies(states=states, internals=internals, auxiliaries=auxiliaries)

        for name, spec, entropy in util.zip_items(self.actions_spec, entropies):
            entropies[name] = tf.reshape(
                tensor=entropy, shape=(-1, util.product(xs=spec['shape']))
            )

        entropy = tf.concat(values=tuple(entropies.values()), axis=1)
        if mean:
            entropy = tf.math.reduce_mean(input_tensor=entropy, axis=1)

        return entropy

    def tf_kl_divergence(self, states, internals, auxiliaries, other=None, mean=True):
        kl_divergences = self.kl_divergences(
            states=states, internals=internals, auxiliaries=auxiliaries, other=other
        )

        for name, spec, kl_divergence in util.zip_items(self.actions_spec, kl_divergences):
            kl_divergences[name] = tf.reshape(
                tensor=kl_divergence, shape=(-1, util.product(xs=spec['shape']))
            )

        kl_divergence = tf.concat(values=tuple(kl_divergences.values()), axis=1)
        if mean:
            kl_divergence = tf.math.reduce_mean(input_tensor=kl_divergence, axis=1)

        return kl_divergence

    def tf_sample_actions(self, states, internals, auxiliaries, deterministic, return_internals):
        raise NotImplementedError

    def tf_log_probabilities(self, states, internals, auxiliaries, actions):
        raise NotImplementedError

    def tf_entropies(self, states, internals, auxiliaries):
        raise NotImplementedError

    def tf_kl_divergences(self, states, internals, auxiliaries, other=None):
        raise NotImplementedError
