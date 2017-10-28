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
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorforce import util
from tensorforce.models import DistributionModel
from tensorforce.core.networks import Network
from tensorforce.core.optimizers import Synchronization


class QModel(DistributionModel):
    """
    Q-value model.
    """

    def __init__(self, states_spec, actions_spec, network_spec, config):

        with tf.name_scope(name=config.scope):
            # Target network
            self.target_network = Network.from_spec(spec=network_spec, kwargs=dict(scope='target'))

            # Target network optimizer
            self.target_optimizer = Synchronization(
                sync_frequency=config.target_sync_frequency,
                update_weight=config.target_update_weight
            )

        self.double_q_model = config.double_q_model

        assert config.huber_loss is None or config.huber_loss > 0.0
        self.huber_loss = config.huber_loss

        super(QModel, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=network_spec,
            config=config
        )

    def tf_q_value(self, embedding, distr_params, action, name):
        # Mainly for NAF.
        return self.distributions[name].state_action_value(distr_params=distr_params, action=action)

    def tf_q_delta(self, q_value, next_q_value, terminal, reward):
        """
        Creates the deltas (or advantage) of the Q values.

        :return: A list of deltas per action
        """
        for _ in range(util.rank(q_value) - 1):
            terminal = tf.expand_dims(input=terminal, axis=1)
            reward = tf.expand_dims(input=reward, axis=1)

        multiples = (1,) + util.shape(q_value)[1:]
        terminal = tf.tile(input=terminal, multiples=multiples)
        reward = tf.tile(input=reward, multiples=multiples)

        zeros = tf.zeros_like(tensor=next_q_value)
        next_q_value = tf.where(condition=terminal, x=zeros, y=(self.discount * next_q_value))

        return reward + next_q_value - q_value  # tf.stop_gradient(q_target)

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward):
        embedding = self.network.apply(
            x={name: state[:-1] for name, state in states.items()},
            internals=[internal[:-1] for internal in internals])

        # Both networks can use the same internals, could that be a problem?
        # Otherwise need to handle internals indices correctly everywhere
        target_embedding = self.target_network.apply(
            x={name: state[1:] for name, state in states.items()},
            internals=[internal[1:] for internal in internals]
        )

        deltas = list()
        for name, distribution in self.distributions.items():

            distr_params = distribution.parameters(x=embedding)
            target_distr_params = distribution.parameters(x=target_embedding)  # TODO: separate distribution parameters?

            q_value = self.tf_q_value(embedding=embedding, distr_params=distr_params, action=actions[name][:-1], name=name)

            if self.double_q_model:
                action_taken = distribution.sample(distr_params=distr_params, deterministic=True)
            else:
                action_taken = distribution.sample(distr_params=target_distr_params, deterministic=True)

            next_q_value = distribution.state_action_value(distr_params=target_distr_params, action=action_taken)

            delta = self.tf_q_delta(q_value=q_value, next_q_value=next_q_value, terminal=terminal[:-1], reward=reward[:-1])

            collapsed_size = util.prod(util.shape(delta)[1:])
            delta = tf.reshape(tensor=delta, shape=(-1, collapsed_size))

            deltas.append(delta)

        # Surrogate loss as the mean squared error between actual observed rewards and expected rewards
        loss_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=deltas, axis=1), axis=1)

        # Optional Huber loss
        if self.huber_loss is not None and self.huber_loss > 0.0:
            return tf.where(
                condition=(tf.abs(x=loss_per_instance) <= self.huber_loss),
                x=(0.5 * tf.square(x=loss_per_instance)),
                y=(self.huber_loss * (tf.abs(x=loss_per_instance) - 0.5 * self.huber_loss))
            )
        else:
            return tf.square(x=loss_per_instance)

    def tf_optimization(self, states, internals, actions, terminal, reward):
        optimization = super(QModel, self).tf_optimization(states, internals, actions, terminal, reward)

        target_optimization = self.target_optimizer.minimize(
            time=self.timestep,
            variables=self.target_network.get_variables(),
            source_variables=self.network.get_variables()
        )

        return tf.group(optimization, target_optimization)

    def get_variables(self, include_non_trainable=False):
        model_variables = super(QModel, self).get_variables(include_non_trainable=include_non_trainable)

        if include_non_trainable:
            # Target network and optimizer variables only included if 'include_non_trainable' set
            target_variables = self.target_network.get_variables(include_non_trainable=include_non_trainable)

            target_optimizer_variables = self.target_optimizer.get_variables()

            return model_variables + target_variables + target_optimizer_variables

        else:
            return model_variables

    def get_summaries(self):
        return super(QModel, self).get_summaries() + self.target_network.get_summaries()
