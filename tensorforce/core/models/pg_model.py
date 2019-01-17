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

from tensorforce.core import baseline_modules, Module, optimizer_modules, parameter_modules
from tensorforce.core.models import DistributionModel


class PGModel(DistributionModel):
    """
    Base class for policy gradient models. It optionally defines a baseline
    and handles its optimization. It implements the `tf_loss_per_instance` function, but requires
    subclasses to implement `tf_pg_loss_per_instance`.
    """

    def __init__(
        self,
        # Model
        states, actions, scope, device, saver, summarizer, execution, parallel_interactions,
        buffer_observe, exploration, variable_noise, states_preprocessing, reward_preprocessing,
        # MemoryModel
        update_mode, memory, optimizer, discount,
        # DistributionModel
        network, distributions, entropy_regularization,
        # PGModel
        baseline_mode, baseline, baseline_optimizer, gae_lambda
    ):
        super().__init__(
            # Model
            states=states, actions=actions, scope=scope, device=device, saver=saver,
            summarizer=summarizer, execution=execution,
            parallel_interactions=parallel_interactions, buffer_observe=buffer_observe,
            exploration=exploration, variable_noise=variable_noise,
            states_preprocessing=states_preprocessing, reward_preprocessing=reward_preprocessing,
            # MemoryModel
            update_mode=update_mode, memory=memory, optimizer=optimizer, discount=discount,
            # DistributionModel
            network=network, distributions=distributions,
            entropy_regularization=entropy_regularization, requires_deterministic=False
        )

        # Baseline mode
        assert baseline_mode is None or baseline_mode in ('states', 'network')
        self.baseline_mode = baseline_mode

        # Baseline
        if baseline is None:
            assert self.baseline_mode is None
        elif all(name in self.states_spec for name in baseline):
            # Implies AggregatedBaseline
            assert self.baseline_mode == 'states'
            inputs_spec = OrderedDict()
            for name, spec in self.states_spec.items():
                inputs_spec[name] = dict(spec)
                inputs_spec[name]['batched'] = True
            self.baseline = self.add_module(
                name='baseline', module='aggregated', modules=baseline_modules,
                is_trainable=(baseline_optimizer is None), is_subscope=True, baselines=baseline,
                inputs_spec=inputs_spec
            )
        else:
            assert self.baseline_mode is not None
            if self.baseline_mode == 'states':
                inputs_spec = OrderedDict()
                for name, spec in self.states_spec.items():
                    inputs_spec[name] = dict(spec)
                    inputs_spec[name]['batched'] = True
            elif self.baseline_mode == 'network':
                inputs_spec = self.network.get_output_spec()
            self.baseline = self.add_module(
                name='baseline', module=baseline, modules=baseline_modules,
                is_trainable=(baseline_optimizer is None), is_subscope=True,
                inputs_spec=inputs_spec
            )

        # Baseline optimizer
        if baseline_optimizer is None:
            self.baseline_optimizer = None
        else:
            assert self.baseline_mode is not None
            self.baseline_optimizer = self.add_module(
                name='baseline-optimizer', module=baseline_optimizer, modules=optimizer_modules
            )

        # Generalized advantage function
        assert gae_lambda is None or self.baseline_mode is not None
        if gae_lambda is None:
            self.gae_lambda = None
        else:
            self.gae_lambda = self.add_module(
                name='gae-lambda', module=gae_lambda, modules=parameter_modules, dtype='float'
            )

        # TODO: Baseline internal states !!! (see target_network q_model)

    def as_local_model(self):
        super().as_local_model()

        if self.baseline_optimizer_spec is not None:
            self.baseline_optimizer_spec = dict(
                type='global_optimizer',
                optimizer=self.baseline_optimizer_spec
            )

    def tf_reward_estimation(self, states, internals, terminal, reward):
        if self.baseline_mode is None:
            return self.discounted_cumulative_reward(terminal=terminal, reward=reward)

        else:
            if self.baseline_mode == 'states':
                state_value = self.baseline.predict(states=states, internals=internals)

            elif self.baseline_mode == 'network':
                embedding = self.network.apply(x=states, internals=internals)
                state_value = self.baseline.predict(
                    states=tf.stop_gradient(input=embedding), internals=internals
                )

            if self.gae_lambda is None:
                reward = self.discounted_cumulative_reward(terminal=terminal, reward=reward)
                advantage = reward - state_value

            else:
                next_state_value = tf.concat(values=(state_value[1:], (0.0,)), axis=0)
                zeros = tf.zeros_like(tensor=next_state_value)
                next_state_value = tf.where(condition=terminal, x=zeros, y=next_state_value)
                discount = self.discount.value()
                td_residual = reward + discount * next_state_value - state_value
                gae_lambda = self.gae_lambda.value()
                advantage = self.discounted_cumulative_reward(
                    terminal=terminal, reward=td_residual, discount=(discount * gae_lambda)
                )

            # Normalize advantage.
            # mean, variance = tf.nn.moments(advantage, axes=[0], keep_dims=True)
            # advantage = (advantage - mean) / tf.sqrt(x=tf.maximum(x=variance, y=util.epsilon))

            return advantage

    def tf_regularization_losses(self, states, internals):
        losses = super().tf_regularization_losses(states=states, internals=internals)

        if self.baseline_mode is not None and self.baseline_optimizer is None:
            baseline_regularization_loss = self.baseline.regularization_loss()
            if baseline_regularization_loss is not None:
                losses['baseline'] = baseline_regularization_loss

        return losses

    def tf_baseline_loss(self, states, internals, reward, reference=None):
        """
        Creates the TensorFlow operations for calculating the baseline loss of a batch.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            reward: Reward tensor.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss tensor.
        """
        Module.update_tensors(**states, **internals, reward=reward)
        if self.baseline_mode == 'states':
            loss = self.baseline.total_loss(states=states, internals=internals, reward=reward)

        elif self.baseline_mode == 'network':
            states = self.network.apply(x=states, internals=internals)
            loss = self.baseline.total_loss(states=states, internals=internals, reward=reward)

        return loss

    def baseline_optimizer_arguments(self, states, internals, reward):
        """
        Returns the baseline optimizer arguments including the time, the list of variables to  
        optimize, and various functions which the optimizer might require to perform an update  
        step.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            reward: Reward tensor.

        Returns:
            Baseline optimizer arguments as dict.
        """
        arguments = dict(
            time=self.global_timestep, variables=self.baseline.get_variables(),
            arguments=dict(states=states, internals=internals, reward=reward),
            fn_reference=self.baseline.reference, fn_loss=self.baseline_loss,
            # source_variables=self.network.get_variables()
        )
        if self.global_model is not None:
            arguments['global_variables'] = self.global_model.baseline.get_variables()
        return arguments

    def tf_optimization(self, states, internals, actions, terminal, reward, next_states=None, next_internals=None):
        assert next_states is None and next_internals is None  # temporary

        estimated_reward = self.reward_estimation(
            states=states, internals=internals, terminal=terminal, reward=reward
        )
        if self.baseline_optimizer is not None:
            estimated_reward = tf.stop_gradient(input=estimated_reward)

        optimization = super().tf_optimization(
            states=states, internals=internals, actions=actions, terminal=terminal,
            reward=estimated_reward, next_states=next_states, next_internals=next_internals
        )

        if self.baseline_optimizer is not None:
            cumulative_reward = self.discounted_cumulative_reward(terminal=terminal, reward=reward)

            arguments = self.baseline_optimizer_arguments(
                states=states,
                internals=internals,
                reward=cumulative_reward,
            )
            baseline_optimization = self.baseline_optimizer.minimize(**arguments)

            optimization = tf.group(optimization, baseline_optimization)

        return optimization
