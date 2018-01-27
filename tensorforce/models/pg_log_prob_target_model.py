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

from tensorforce.core.baselines import Baseline, AggregatedBaseline
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
        summarizer,
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
        self.target_baseline = None
        self.target_baseline_optimizer = None

        super(PGLogProbModel, self).__init__(
            states=states,
            actions=actions,
            scope=scope,
            device=device,
            saver=saver,
            summarizer=summarizer,
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

        # Target baseline
        if self.baseline_mode:
            if all(name in self.states_spec for name in self.baseline_spec):
                # Implies AggregatedBaseline.
                assert self.baseline_mode == 'states'
                self.target_baseline = AggregatedBaseline(baselines=self.baseline_spec)
            else:
                self.target_baseline = Baseline.from_spec(
                    spec=self.baseline_spec,
                    kwargs=dict(
                        summary_labels=self.summary_labels,
                        scope='target_baseline'
                    )
                )

            # Target baseline optimizer
            self.target_baseline_optimizer = Synchronization(
                sync_frequency=self.target_sync_frequency,
                update_weight=self.target_update_weight
            )

    def tf_reward_estimation(self, states, internals, terminal, reward, update):
        if self.baseline_mode is None:
            reward = self.fn_discounted_cumulative_reward(terminal=terminal, reward=reward, discount=self.discount)

        else:
            assert self.target_baseline
            if self.baseline_mode == 'states':
                state_value = self.target_baseline.predict(
                    states=states,
                    internals=internals,
                    update=update
                )

            elif self.baseline_mode == 'network':
                embedding = self.target_network.apply(
                    x=states,
                    internals=internals,
                    update=update
                )
                state_value = self.target_baseline.predict(
                    states=embedding,
                    internals=internals,
                    update=update
                )

            if self.gae_lambda is None:
                reward = self.fn_discounted_cumulative_reward(
                    terminal=terminal,
                    reward=reward,
                    discount=self.discount
                )
                reward -= state_value

            else:
                next_state_value = tf.concat(values=(state_value[1:], (0.0,)), axis=0)
                zeros = tf.zeros_like(tensor=next_state_value)
                next_state_value = tf.where(condition=terminal, x=zeros, y=next_state_value)
                td_residual = reward + self.discount * next_state_value - state_value
                gae_discount = self.discount * self.gae_lambda
                reward = self.fn_discounted_cumulative_reward(
                    terminal=terminal,
                    reward=td_residual,
                    discount=gae_discount
                )

        return reward

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

        if self.target_baseline:
            target_baseline_optimization = self.target_baseline_optimizer.minimize(
                time=self.timestep,
                variables=self.target_baseline.get_variables(),
                source_variables=self.baseline.get_variables()
            )
            return tf.group(optimization, target_optimization, target_baseline_optimization)

        return tf.group(optimization, target_optimization)

    def get_variables(self, include_non_trainable=False):
        model_variables = super(PGLogProbModel, self).get_variables(include_non_trainable=include_non_trainable)

        if include_non_trainable:
            # Target network and optimizer variables only included if 'include_non_trainable' set
            target_variables = self.target_network.get_variables(include_non_trainable=include_non_trainable)
            target_distributions_variables = self.get_distributions_variables(self.target_distributions)
            target_optimizer_variables = self.target_optimizer.get_variables()

            if self.target_baseline:
                target_baseline_variables = self.target_baseline.get_variables()
                return model_variables + target_variables + target_optimizer_variables + \
                    target_distributions_variables + target_baseline_variables

            return model_variables + target_variables + target_optimizer_variables + target_distributions_variables
        else:
            return model_variables

    def get_summaries(self):
        target_distributions_summaries = self.get_distributions_summaries(self.target_distributions)
        return super(PGLogProbModel, self).get_summaries() + self.target_network.get_summaries() \
            + target_distributions_summaries
