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

from tensorforce.core.models import DistributionModel

from tensorforce.core.networks import Network
from tensorforce.core.optimizers import Optimizer, Synchronization


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

        self.critic_network = None
        self.critic_optimizer = None
        self.target_critic_network = None
        self.target_critic_optimizer = None

        super().__init__(
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
        assert self.requires_deterministic

    def setup_components_and_tf_funcs(self, custom_getter=None):
        custom_getter = super().setup_components_and_tf_funcs(custom_getter)

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

        # critic
        self.critic_network = Network.from_spec(
            spec=self.critic_network_spec,
            kwargs=dict(scope='critic')
        )

        self.target_critic_network = Network.from_spec(
            spec=self.critic_network_spec,
            kwargs=dict(scope='target-critic')
        )

        self.critic_optimizer = Optimizer.from_spec(
            spec=self.critic_optimizer_spec,
            kwargs=dict(summary_labels=self.summary_labels)
        )

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
        return custom_getter

    def tf_target_actions_and_internals(self, states, internals, deterministic=True):
        embedding, internals = self.target_network.apply(
            x=states,
            internals=internals,
            update=tf.constant(value=False),
            return_internals=True
        )

        actions = dict()
        for name in sorted(self.target_distributions):
            distribution = self.target_distributions[name]
            distr_params = distribution.parameterize(x=embedding)
            actions[name] = distribution.sample(
                distr_params=distr_params,
                deterministic=tf.logical_or(x=deterministic, y=self.requires_deterministic)
            )

        return actions, internals

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward, next_states, next_internals, update, reference=None):
        states_actions = dict(states)
        states_actions.update(actions)
        q = self.critic_network.apply(x=states_actions, internals=internals, update=update)
        return -q

    def tf_predict_target_q(self, states, internals, terminal, actions, reward, update):
        states_actions = dict(states)
        states_actions.update(actions)
        q_value = self.target_critic_network.apply(x=states_actions, internals=internals, update=update)
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

        states_actions = dict(states)
        states_actions.update(actions)
        real_q = self.critic_network.apply(x=states_actions, internals=internals, update=update)

        # Update critic
        def fn_critic_loss(predicted_q, real_q):
            return tf.reduce_mean(tf.square(real_q - predicted_q))

        critic_optimization = self.critic_optimizer.minimize(
            time=self.timestep,
            variables=self.critic_network.get_variables(),
            arguments=dict(
                predicted_q=predicted_q,
                real_q=real_q
            ),
            fn_loss=fn_critic_loss)

        # Update actor
        predicted_actions, predicted_internals = self.fn_actions_and_internals(
            states=states, internals=internals, deterministic=True
        )

        optimization = super().tf_optimization(
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
            variables=self.target_critic_network.get_variables(),
            source_variables=self.critic_network.get_variables()
        )

        return tf.group(critic_optimization, optimization, target_optimization, target_critic_optimization)

    def get_variables(self, include_submodules=False, include_nontrainable=False):
        model_variables = super().get_variables(
            include_submodules=include_submodules,
            include_nontrainable=include_nontrainable
        )
        critic_variables = self.critic_network.get_variables(include_nontrainable=include_nontrainable)
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

            target_critic_variables = self.target_critic_network.get_variables(include_nontrainable=include_nontrainable)
            model_variables += target_critic_variables

            if include_nontrainable:
                target_optimizer_variables = self.target_network_optimizer.get_variables()
                model_variables += target_optimizer_variables

                target_critic_optimizer_variables = self.target_critic_optimizer.get_variables()
                model_variables += target_critic_optimizer_variables

        return model_variables

    def get_components(self):
        result = dict(super().get_components())
        result[DPGTargetModel.COMPONENT_CRITIC] = self.critic_network
        result[DPGTargetModel.COMPONENT_TARGET_NETWORK] = self.target_network
        for name in sorted(self.target_distributions):
            result["%s_%s" % (DPGTargetModel.COMPONENT_TARGET_DISTRIBUTION, name)] = self.target_distributions[name]
        if len(self.target_distributions) == 1:
            result[DPGTargetModel.COMPONENT_TARGET_DISTRIBUTION] = self.target_distributions[next(iter(sorted(self.target_distributions)))]
        return result
