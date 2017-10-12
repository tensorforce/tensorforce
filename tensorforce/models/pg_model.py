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


"""
The `PGModel` class implements the specified reward estimation. It optionally defines a baseline
and handles its optimization. It implements the `tf_loss_per_instance` function, but requires
subclasses to implement `tf_pg_loss_per_instance`.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorforce.core.baselines import Baseline, AggregatedBaseline
from tensorforce.core.optimizers import Optimizer
from tensorforce.models import DistributionModel


class PGModel(DistributionModel):
    """Base class for policy gradient models."""

    def __init__(self, states_spec, actions_spec, network_spec, config):
        # Baseline mode
        assert config.baseline_mode is None or config.baseline_mode in ('states', 'network')
        self.baseline_mode = config.baseline_mode

        with tf.name_scope(name=config.scope):
            # Baseline
            if config.baseline is None:
                assert self.baseline_mode is None
                self.baseline = None

            elif all(name in states_spec for name in config.baseline):
                # Implies AggregatedBaseline
                assert self.baseline_mode == 'states'
                self.baseline = AggregatedBaseline(baselines=config.baseline)

            else:
                assert self.baseline_mode is not None
                self.baseline = Baseline.from_spec(spec=config.baseline)

            # Baseline optimizer
            if config.baseline_optimizer is None:
                self.baseline_optimizer = None
            else:
                assert self.baseline_mode is not None
                self.baseline_optimizer = Optimizer.from_spec(spec=config.baseline_optimizer)

        # Generalized advantage function
        assert config.gae_lambda is None or (0.0 <= config.gae_lambda <= 1.0 and self.baseline_mode is not None)
        self.gae_lambda = config.gae_lambda

        super(PGModel, self).__init__(states_spec, actions_spec, network_spec, config)

    def initialize(self, custom_getter):
        super(PGModel, self).initialize(custom_getter)

        # PG loss per instance function
        self.fn_pg_loss_per_instance = tf.make_template(
            name_='pg-loss-per-instance',
            func_=self.tf_pg_loss_per_instance,
            create_scope_now_=True,
            custom_getter_=custom_getter
        )

    def tf_pg_loss_per_instance(self, states, internals, actions, terminal, reward):
        """
        Creates the TensorFlow operations for calculating the (policy-gradient-specific) loss per batch
        instance of the given input states and actions, after the specified reward/advantage calculations.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward):
        if self.baseline_mode is None:
            reward = self.fn_discounted_cumulative_reward(reward=reward, terminal=terminal, discount=self.discount)

        else:
            if self.baseline_mode == 'states':
                state_value = self.baseline.predict(states=states)
            elif self.baseline_mode == 'network':
                embedding = self.network.apply(x=states, internals=internals)
                state_value = self.baseline.predict(states=embedding)

            if self.gae_lambda is None:
                reward = self.fn_discounted_cumulative_reward(reward=reward, terminal=terminal, discount=self.discount)
                reward -= state_value

            else:
                state_value_change = self.discount * state_value[1:] - state_value[:-1]
                state_value_change = tf.concat(values=(state_value_change, (0.0,)), axis=0)
                zeros = tf.zeros_like(tensor=state_value_change)
                state_value_change = tf.where(condition=terminal, x=zeros, y=state_value_change)
                td_residual = reward + state_value_change
                gae_discount = self.discount * self.gae_lambda
                self.fn_discounted_cumulative_reward(reward=td_residual, terminal=terminal, discount=gae_discount)

        return self.fn_pg_loss_per_instance(states=states, internals=internals, actions=actions, terminal=terminal, reward=reward)

    def tf_optimization(self, states, internals, actions, terminal, reward):
        optimization = super(PGModel, self).tf_optimization(states, internals, actions, terminal, reward)

        if self.baseline_mode is None:
            return optimization

        reward = self.fn_discounted_cumulative_reward(reward=reward, terminal=terminal, discount=self.discount)

        if self.baseline_mode == 'states':
            fn_loss = (lambda: self.baseline.loss(states=states, reward=reward))
        elif self.baseline_mode == 'network':
            fn_loss = (lambda: self.baseline.loss(states=self.network.apply(x=states, internals=internals), reward=reward))

        # TODO: time as argument?
        baseline_optimization = self.baseline_optimizer.minimize(
            time=self.time,
            variables=self.baseline.get_variables(), fn_loss=fn_loss, source_variables=self.network.get_variables())

        return tf.group(optimization, baseline_optimization)

    def get_variables(self):
        if self.baseline_mode is not None and self.baseline_optimizer is None:
            return super(PGModel, self).get_variables() + self.baseline.get_variables()
        else:
            return super(PGModel, self).get_variables()
