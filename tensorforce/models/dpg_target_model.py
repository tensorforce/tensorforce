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

from tensorforce import util, TensorForceError
from tensorforce.models import DistributionModel

from tensorforce.core.networks import Network, LayerBasedNetwork, Dense, Linear, TFLayer, Nonlinearity
from tensorforce.core.optimizers import Optimizer, Synchronization


class DDPGCriticNetwork(LayerBasedNetwork):
    def __init__(self, scope='ddpg-critic-network', summary_labels=(), size_t0=400, size_t1=300):
        super(DDPGCriticNetwork, self).__init__(scope=scope, summary_labels=summary_labels)

        self.t0l = Linear(size=size_t0, scope='linear0')
        self.t0b = TFLayer(layer='batch_normalization', scope='batchnorm0', center=True, scale=True)
        self.t0n = Nonlinearity(name='relu', scope='relu0')

        self.t1l = Linear(size=size_t1, scope='linear1')
        self.t1b = TFLayer(layer='batch_normalization', scope='batchnorm1', center=True, scale=True)
        self.t1n = Nonlinearity(name='relu', scope='relu1')

        self.t2d = Dense(size=1, activation='tanh', scope='dense0',
                         weights=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

        self.add_layer(self.t0l)
        self.add_layer(self.t0b)
        self.add_layer(self.t0n)

        self.add_layer(self.t1l)
        self.add_layer(self.t1b)
        self.add_layer(self.t1n)

        self.add_layer(self.t2d)

    def tf_apply(self, x, internals, update, return_internals=False):
        assert x['states'], x['actions']

        if isinstance(x['states'], dict):
            if len(x['states']) != 1:
                raise TensorForceError('DDPG critic network must have only one state input, but {} given.'.format(
                    len(x['states'])))
            x_states = next(iter(x['states'].values()))
        else:
            x_states = x['states']

        if isinstance(x['actions'], dict):
            if len(x['actions']) != 1:
                raise TensorForceError('DDPG critic network must have only one action input, but {} given.'.format(
                    len(x['actions'])))
            x_actions = next(iter(x['actions'].values()))
        else:
            x_actions = x['actions']

        x_actions = tf.reshape(tf.cast(x_actions, dtype=tf.float32), (-1, 1))

        out = self.t0l.apply(x=x_states, update=update)
        out = self.t0b.apply(x=out, update=update)
        out = self.t0n.apply(x=out, update=update)

        out = self.t1l.apply(x=tf.concat([out, x_actions], axis=-1), update=update)
        out = self.t1b.apply(x=out, update=update)
        out = self.t1n.apply(x=out, update=update)

        out = self.t2d.apply(x=out, update=update)

        # Remove last dimension because we only return Q values for one state and action
        out = tf.squeeze(out)

        if return_internals:
            # Todo: Internals management
            return out, None
        else:
            return out


class DPGTargetModel(DistributionModel):
    """
    Policy gradient model log likelihood model with target network (e.g. DDPG)
    """

    COMPONENT_CRITIC = "critic"
    COMPONENT_TARGET_NETWORK = "target_network"
    COMPONENT_TARGET_DISTRIBUTION = "target_distribution"

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
        critic_network,
        critic_optimizer,
        target_sync_frequency,
        target_update_weight
    ):

        self.critic_network_spec = critic_network
        self.critic_optimizer_spec = critic_optimizer

        self.target_sync_frequency = target_sync_frequency
        self.target_update_weight = target_update_weight

        # self.network is the actor, self.critic is the critic
        self.target_network = None
        self.target_network_optimizer = None

        self.critic = None
        self.critic_optimizer = None
        self.target_critic = None
        self.target_critic_optimizer = None

        super(DPGTargetModel, self).__init__(
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
            discount=discount,
            network=network,
            distributions=distributions,
            entropy_regularization=entropy_regularization,
            requires_deterministic=True
        )

        assert self.memory_spec["include_next_states"]
        assert self.requires_deterministic == True

    def initialize(self, custom_getter):
        super(DPGTargetModel, self).initialize(custom_getter)

        # Target network
        self.target_network = Network.from_spec(
            spec=self.network_spec,
            kwargs=dict(scope='target-network', summary_labels=self.summary_labels)
        )

        # Target network optimizer
        self.target_network_optimizer = Synchronization(
            sync_frequency=self.target_sync_frequency,
            update_weight=self.target_update_weight
        )

        # Target network distributions
        self.target_distributions = self.create_distributions()

        # Critic
        size_t0 = self.critic_network_spec['size_t0']
        size_t1 = self.critic_network_spec['size_t1']

        self.critic = DDPGCriticNetwork(scope='critic', size_t0=size_t0, size_t1=size_t1)
        self.critic_optimizer = Optimizer.from_spec(
            spec=self.critic_optimizer_spec,
            kwargs=dict(summary_labels=self.summary_labels)
        )

        self.target_critic = DDPGCriticNetwork(scope='target-critic', size_t0=size_t0, size_t1=size_t1)

        # Target critic optimizer
        self.target_critic_optimizer = Synchronization(
            sync_frequency=self.target_sync_frequency,
            update_weight=self.target_update_weight
        )

        self.fn_target_actions_and_internals = tf.make_template(
            name_='target-actions-and-internals',
            func_=self.tf_target_actions_and_internals,
            custom_getter_=custom_getter
        )

        self.fn_predict_target_q = tf.make_template(
            name_='predict-target-q',
            func_=self.tf_predict_target_q,
            custom_getter_=custom_getter
        )

    def tf_target_actions_and_internals(self, states, internals, deterministic=True):
        embedding, internals = self.target_network.apply(
            x=states,
            internals=internals,
            update=tf.constant(value=False),
            return_internals=True
        )

        actions = dict()
        for name, distribution in self.target_distributions.items():
            distr_params = distribution.parameterize(x=embedding)
            actions[name] = distribution.sample(
                distr_params=distr_params,
                deterministic=tf.logical_or(x=deterministic, y=self.requires_deterministic)
            )

        return actions, internals

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward, next_states, next_internals, update, reference=None):
        q = self.critic.apply(dict(states=states, actions=actions), internals=internals, update=update)
        return -q

    def tf_predict_target_q(self, states, internals, terminal, actions, reward, update):
        q_value = self.target_critic.apply(dict(states=states, actions=actions), internals=internals, update=update)
        return reward + (1. - tf.cast(terminal, dtype=tf.float32)) * self.discount * q_value

    def tf_optimization(self, states, internals, actions, terminal, reward, next_states=None, next_internals=None):
        update = tf.constant(value=True)

        # Predict actions from target actor
        next_target_actions, next_target_internals = self.fn_target_actions_and_internals(
            states=next_states, internals=next_internals, deterministic=True
        )

        # Predicted Q value of next states
        predicted_q = self.fn_predict_target_q(
            states=next_states, internals=next_internals, actions=next_target_actions, terminal=terminal,
            reward=reward, update=update
        )

        predicted_q = tf.stop_gradient(input=predicted_q)

        real_q = self.critic.apply(dict(states=states, actions=actions), internals=internals, update=update)

        # Update critic
        def fn_critic_loss(predicted_q, real_q):
            return tf.reduce_mean(tf.square(real_q - predicted_q))

        critic_optimization = self.critic_optimizer.minimize(
            time=self.timestep,
            variables=self.critic.get_variables(),
            arguments=dict(
                predicted_q=predicted_q,
                real_q=real_q
            ),
            fn_loss=fn_critic_loss)

        # Update actor
        predicted_actions, predicted_internals = self.fn_actions_and_internals(
            states=states, internals=internals, deterministic=True
        )

        optimization = super(DPGTargetModel, self).tf_optimization(
            states=states,
            internals=internals,
            actions=predicted_actions,
            terminal=terminal,
            reward=reward,
            next_states=next_states,
            next_internals=next_internals
        )

        # Update target actor (network) and critic
        network_distributions_variables = [
            variable for name in sorted(self.distributions)
            for variable in self.distributions[name].get_variables(include_nontrainable=False)
        ]

        target_distributions_variables = [
            variable for name in sorted(self.target_distributions)
            for variable in self.target_distributions[name].get_variables(include_nontrainable=False)
        ]

        target_optimization = self.target_network_optimizer.minimize(
            time=self.timestep,
            variables=self.target_network.get_variables() + target_distributions_variables,
            source_variables=self.network.get_variables() + network_distributions_variables
        )

        target_critic_optimization = self.target_critic_optimizer.minimize(
            time=self.timestep,
            variables=self.target_critic.get_variables(),
            source_variables=self.critic.get_variables()
        )

        return tf.group(critic_optimization, optimization, target_optimization, target_critic_optimization)

    def get_variables(self, include_submodules=False, include_nontrainable=False):
        model_variables = super(DPGTargetModel, self).get_variables(
            include_submodules=include_submodules,
            include_nontrainable=include_nontrainable
        )
        critic_variables = self.critic.get_variables(include_nontrainable=include_nontrainable)
        model_variables += critic_variables

        if include_nontrainable:
            critic_optimizer_variables = self.critic_optimizer.get_variables()

            for variable in critic_optimizer_variables:
                if variable in model_variables:
                    model_variables.remove(variable)

            model_variables += critic_optimizer_variables

        if include_submodules:
            target_variables = self.target_network.get_variables(include_nontrainable=include_nontrainable)
            model_variables += target_variables

            target_distributions_variables = [
                variable for name in sorted(self.target_distributions)
                for variable in self.target_distributions[name].get_variables(include_nontrainable=include_nontrainable)
            ]
            model_variables += target_distributions_variables

            target_critic_variables = self.target_critic.get_variables(include_nontrainable=include_nontrainable)
            model_variables += target_critic_variables

            if include_nontrainable:
                target_optimizer_variables = self.target_network_optimizer.get_variables()
                model_variables += target_optimizer_variables

                target_critic_optimizer_variables = self.target_critic_optimizer.get_variables()
                model_variables += target_critic_optimizer_variables

        return model_variables

    def get_summaries(self):
        target_network_summaries = self.target_network.get_summaries()
        target_distributions_summaries = [
            summary for name in sorted(self.target_distributions)
            for summary in self.target_distributions[name].get_summaries()
        ]

        # Todo: Critic summaries
        return super(DPGTargetModel, self).get_summaries() + target_network_summaries \
            + target_distributions_summaries

    def get_components(self):
        result = dict(super(DPGTargetModel, self).get_components())
        result[DPGTargetModel.COMPONENT_CRITIC] = self.critic
        result[DPGTargetModel.COMPONENT_TARGET_NETWORK] = self.target_network
        for action, distribution in self.target_distributions.items():
            result["%s_%s" % (DPGTargetModel.COMPONENT_TARGET_DISTRIBUTION, action)] = distribution
        if len(self.target_distributions) == 1:
            result[DPGTargetModel.COMPONENT_TARGET_DISTRIBUTION] = next(iter(self.target_distributions.values()))
        return result
