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
    Base class for Q-value models
    """

    def __init__(self, states_spec, actions_spec, network_spec, config):
        # Target network
        self.target_network = Network.from_spec(spec=network_spec, kwargs=dict(scope='target'))

        # Target network optimizer
        self.target_optimizer = Synchronization(
            update_frequency=config.target_update_frequency,
            update_weight=config.update_target_weight
        )

        self.double_dqn = config.double_dqn
        self.huber_loss = config.huber_loss

        super(QModel, self).__init__(states_spec, actions_spec, network_spec, config)

        # self.last_target_update = 0
        # Synchronize target with training network
        # self.maybe_update_target(force=True)

    def initialize(self, custom_getter):
        super(QModel, self).initialize(custom_getter)

        # Target network internals
        self.internal_inputs.extend(self.target_network.internal_inputs())
        self.internal_inits.extend(self.target_network.internal_inits())

    def tf_q_value(self, logits, action):
        raise NotImplementedError

    def tf_q_delta(self, q_value, next_q_value, terminal, reward):
        """
        Creates the deltas (or advantage) of the Q values
        :return: A list of deltas per action
        """
        zeros = tf.zeros_like(tensor=next_q_value)
        next_q_value = tf.where(condition=terminal, x=zeros, y=(self.discount * next_q_value))

        delta = reward + next_q_value - q_value  # tf.stop_gradient(q_target)
        collapsed_size = util.prod(util.shape(delta)[1:])
        return tf.reshape(tensor=delta, shape=(-1, collapsed_size))

    def tf_loss_per_instance(self, states, actions, terminal, reward, internals):
        embedding = self.network.apply(x={name: state[:-1] for name, state in states.items()}, internals=internals[:-1])
        target_embedding = self.target_network.apply(x={name: state[1:] for name, state in states.items()}, internals=internals[1:])
        deltas = list()

        for name, distribution in self.distributions.items():
            distr_params = distribution.parameters(x=embedding)
            q_value = distribution.state_action_value(distr_params=distr_params, action=actions[name][:-1])
            # q_value = self.q_value(logits=distr_params[0], action=actions[name])  # !!! really? always [0]?
            target_distr_params = distribution.parameters(x=target_embedding)  # requires a different distribution class?

            if self.double_dqn:
                action_taken = distribution.sample(distr_params=distr_params, deterministic=True)
            else:
                action_taken = distribution.sample(distr_params=target_distr_params, deterministic=True)
            next_q_value = distribution.state_action_value(distr_params=target_distr_params, action=action_taken)
            delta = self.tf_q_delta(q_value=q_value, next_q_value=next_q_value, terminal=terminal[:-1], reward=reward[:-1])
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

    def tf_optimization(self, states, actions, terminal, reward, internals):
        optimization = super(QModel, self).tf_optimization(states, actions, terminal, reward, internals)

        target_optimization = self.target_optimizer.minimize(time=self.time, variables=self.target_network.get_variables(), source_variables=self.network.get_variables())

        return tf.group(optimization, target_optimization)

        # def true_fn():
        #     updates = list()
        #     for v_source, v_target in zip(self.network.get_variables(), self.target_network.get_variables()):
        #         update = v_target.assign_sub(delta=(self.update_target_weight * (v_target - v_source)))
        #         updates.append(update)
        #     update_time = self.last_update.assign(value=self.time)
        #     updates.append(update_time)
        #     return tf.group(*updates)

        # update = (self.time - self.last_update >= self.target_update_frequency)
        # target_optimization = tf.cond(pred=update, true_fn=true_fn, false_fn=tf.no_op)

        # return tf.group(optimization, target_optimization)




    # def get_variables(self):
    #     if False:
    #         return super(QModel, self).get_variables() + self.target_network.get_variables()
    #     else:
    #         return super(QModel, self).get_variables()
