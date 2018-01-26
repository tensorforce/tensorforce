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
from tensorforce.models import PGLogProbModel

from tensorforce.core.networks import Network
from tensorforce.core.optimizers import Synchronization


class PGLogProbTargetModel(PGLogProbModel):
    """
    Policy gradient model log likelihood model with target network (e.g. DDPG)
    """

    def __init__(
        self,
        states,
        actions,
        scope,
        device,
        saver,
        summaries,
        distributed,
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
        baseline_mode,
        baseline,
        baseline_optimizer,
        gae_lambda,
        target_sync_frequency,
        target_update_weight
    ):

        self.target_network_spec = network
        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight

        self.target_network = None
        self.target_optimizer = None
        self.target_distributions = None

        super(PGLogProbModel, self).__init__(
            states=states,
            actions=actions,
            scope=scope,
            device=device,
            saver=saver,
            summaries=summaries,
            distributed=distributed,
            batching_capacity=batching_capacity,
            variable_noise=variable_noise,
            states_preprocessing=states_preprocessing,
            actions_exploration=actions_exploration,
            reward_preprocessing=reward_preprocessing,
            update_mode=update_mode,
            memory=memory,
            optimizer=optimizer,
            discount=discount,
            network=network,
            distributions=distributions,
            entropy_regularization=entropy_regularization,
            baseline_mode=baseline_mode,
            baseline=baseline,
            baseline_optimizer=baseline_optimizer,
            gae_lambda=gae_lambda
        )

    def initialize(self, custom_getter):
        super(PGLogProbTargetModel, self).initialize(custom_getter)

        # Target network
        self.target_network = Network.from_spec(
            spec=self.target_network_spec,
            kwargs=dict(scope='target', summary_labels=self.summary_labels)
        )

        # Target network optimizer
        self.target_optimizer = Synchronization(
            sync_frequency=self.target_sync_frequency,
            update_weight=self.target_update_weight
        )

        # Target network distributions
        self.target_distributions = self.create_distributions()

    def tf_pg_loss_per_instance(self, states, internals, actions, terminal, reward, next_states, next_internals, update):
        embedding = self.target_network.apply(x=states, internals=internals, update=update)
        log_probs = list()

        for name, distribution in self.target_distributions.items():
            distr_params = distribution.parameterize(x=embedding)
            log_prob = distribution.log_probability(distr_params=distr_params, action=actions[name])
            collapsed_size = util.prod(util.shape(log_prob)[1:])
            log_prob = tf.reshape(tensor=log_prob, shape=(-1, collapsed_size))
            log_probs.append(log_prob)
        log_prob = tf.reduce_mean(input_tensor=tf.concat(values=log_probs, axis=1), axis=1)
        return -log_prob * reward

    def tf_optimization(self, states, internals, actions, terminal, reward, next_states=None, next_internals=None):
        optimization = super(PGLogProbModel, self).tf_optimization(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            next_states=next_states,
            next_internals=next_internals
        )

        network_distributions_variables = self.get_distributions_variables(self.distributions)
        target_distributions_variables = self.get_distributions_variables(self.target_distributions)

        target_optimization = self.target_optimizer.minimize(
            time=self.timestep,
            variables=self.target_network.get_variables() + target_distributions_variables,
            source_variables=self.network.get_variables() + network_distributions_variables
        )

        return tf.group(optimization, target_optimization)

    def get_variables(self, include_non_trainable=False):
        model_variables = super(PGLogProbModel, self).get_variables(include_non_trainable=include_non_trainable)

        if include_non_trainable:
            # Target network and optimizer variables only included if 'include_non_trainable' set
            target_variables = self.target_network.get_variables(include_non_trainable=include_non_trainable)
            target_distributions_variables = self.get_distributions_variables(self.target_distributions)
            target_optimizer_variables = self.target_optimizer.get_variables()

            return model_variables + target_variables + target_optimizer_variables + target_distributions_variables
        else:
            return model_variables

    def get_summaries(self):
        target_distributions_summaries = self.get_distributions_summaries(self.target_distributions)
        return super(PGLogProbModel, self).get_summaries() + self.target_network.get_summaries() \
               + target_distributions_summaries
