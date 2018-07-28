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

from tensorforce.core.baselines import Baseline, AggregatedBaseline
from tensorforce.core.optimizers import Optimizer
from tensorforce.models import DistributionModel


class PGModel(DistributionModel):
    """
    Base class for policy gradient models. It optionally defines a baseline
    and handles its optimization. It implements the `tf_loss_per_instance` function, but requires
    subclasses to implement `tf_pg_loss_per_instance`.
    """

    COMPONENT_BASELINE = "baseline"

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
        baseline_mode,
        baseline,
        baseline_optimizer,
        gae_lambda
    ):
        # Baseline mode
        assert baseline_mode is None or baseline_mode in ('states', 'network')
        self.baseline_mode = baseline_mode

        self.baseline_spec = baseline
        self.baseline_optimizer_spec = baseline_optimizer

        # Generalized advantage function
        assert gae_lambda is None or (0.0 <= gae_lambda <= 1.0 and self.baseline_mode is not None)
        self.gae_lambda = gae_lambda

        self.baseline = None
        self.baseline_optimizer = None
        self.fn_reward_estimation = None
        self.fn_baseline_loss = None

        super(PGModel, self).__init__(
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
            requires_deterministic=False
        )

    def as_local_model(self):
        super(PGModel, self).as_local_model()
        if self.baseline_optimizer_spec is not None:
            self.baseline_optimizer_spec = dict(
                type='global_optimizer',
                optimizer=self.baseline_optimizer_spec
            )

    def setup_components_and_tf_funcs(self, custom_getter=None):
        custom_getter = super(PGModel, self).setup_components_and_tf_funcs(custom_getter)

        # Baseline
        if self.baseline_spec is None:
            assert self.baseline_mode is None

        elif all(name in self.states_spec for name in self.baseline_spec):
            # Implies AggregatedBaseline.
            assert self.baseline_mode == 'states'
            self.baseline = AggregatedBaseline(baselines=self.baseline_spec)

        else:
            assert self.baseline_mode is not None
            self.baseline = Baseline.from_spec(
                spec=self.baseline_spec,
                kwargs=dict(
                    summary_labels=self.summary_labels
                )
            )

        # Baseline optimizer
        if self.baseline_optimizer_spec is not None:
            assert self.baseline_mode is not None
            self.baseline_optimizer = Optimizer.from_spec(spec=self.baseline_optimizer_spec)

        # TODO: Baseline internal states !!! (see target_network q_model)

        # Reward estimation
        self.fn_reward_estimation = tf.make_template(
            name_='reward-estimation',
            func_=self.tf_reward_estimation,
            custom_getter_=custom_getter
        )
        # Baseline loss
        self.fn_baseline_loss = tf.make_template(
            name_='baseline-loss',
            func_=self.tf_baseline_loss,
            custom_getter_=custom_getter
        )

        return custom_getter

    def tf_reward_estimation(self, states, internals, terminal, reward, update):
        if self.baseline_mode is None:
            return self.fn_discounted_cumulative_reward(terminal=terminal, reward=reward, discount=self.discount)

        else:
            if self.baseline_mode == 'states':
                state_value = self.baseline.predict(
                    states=states,
                    internals=internals,
                    update=update
                )

            elif self.baseline_mode == 'network':
                embedding = self.network.apply(
                    x=states,
                    internals=internals,
                    update=update
                )
                state_value = self.baseline.predict(
                    states=tf.stop_gradient(input=embedding),
                    internals=internals,
                    update=update
                )

            if self.gae_lambda is None:
                reward = self.fn_discounted_cumulative_reward(
                    terminal=terminal,
                    reward=reward,
                    discount=self.discount
                )
                advantage = reward - state_value

            else:
                next_state_value = tf.concat(values=(state_value[1:], (0.0,)), axis=0)
                zeros = tf.zeros_like(tensor=next_state_value)
                next_state_value = tf.where(condition=terminal, x=zeros, y=next_state_value)
                td_residual = reward + self.discount * next_state_value - state_value
                gae_discount = self.discount * self.gae_lambda
                advantage = self.fn_discounted_cumulative_reward(
                    terminal=terminal,
                    reward=td_residual,
                    discount=gae_discount
                )

            # Normalize advantage.
            # mean, variance = tf.nn.moments(advantage, axes=[0], keep_dims=True)
            # advantage = (advantage - mean) / tf.sqrt(x=tf.maximum(x=variance, y=util.epsilon))

            return advantage

    def tf_regularization_losses(self, states, internals, update):
        losses = super(PGModel, self).tf_regularization_losses(
            states=states,
            internals=internals,
            update=update
        )

        if self.baseline_mode is not None and self.baseline_optimizer is None:
            baseline_regularization_loss = self.baseline.regularization_loss()
            if baseline_regularization_loss is not None:
                losses['baseline'] = baseline_regularization_loss

        return losses

    def tf_baseline_loss(self, states, internals, reward, update, reference=None):
        """
        Creates the TensorFlow operations for calculating the baseline loss of a batch.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss tensor.
        """
        if self.baseline_mode == 'states':
            loss = self.baseline.loss(
                states=states,
                internals=internals,
                reward=reward,
                update=update,
                reference=reference
            )

        elif self.baseline_mode == 'network':
            loss = self.baseline.loss(
                states=self.network.apply(x=states, internals=internals, update=update),
                internals=internals,
                reward=reward,
                update=update,
                reference=reference
            )

        regularization_loss = self.baseline.regularization_loss()
        if regularization_loss is not None:
            loss += regularization_loss

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
            time=self.global_timestep,
            variables=self.baseline.get_variables(),
            arguments=dict(
                states=states,
                internals=internals,
                reward=reward,
                update=tf.constant(value=True),
            ),
            fn_reference=self.baseline.reference,
            fn_loss=self.fn_baseline_loss,
            # source_variables=self.network.get_variables()
        )
        if self.global_model is not None:
            arguments['global_variables'] = self.global_model.baseline.get_variables()
        return arguments

    def tf_optimization(self, states, internals, actions, terminal, reward, next_states=None, next_internals=None):
        assert next_states is None and next_internals is None  # temporary

        estimated_reward = self.fn_reward_estimation(
            states=states,
            internals=internals,
            terminal=terminal,
            reward=reward,
            update=tf.constant(value=True)
        )
        if self.baseline_optimizer is not None:
            estimated_reward = tf.stop_gradient(input=estimated_reward)

        optimization = super(PGModel, self).tf_optimization(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=estimated_reward,
            next_states=next_states,
            next_internals=next_internals
        )

        if self.baseline_optimizer is not None:
            cumulative_reward = self.fn_discounted_cumulative_reward(terminal=terminal, reward=reward, discount=self.discount)

            arguments = self.baseline_optimizer_arguments(
                states=states,
                internals=internals,
                reward=cumulative_reward,
            )
            baseline_optimization = self.baseline_optimizer.minimize(**arguments)

            optimization = tf.group(optimization, baseline_optimization)

        return optimization

    def get_variables(self, include_submodules=False, include_nontrainable=False):
        model_variables = super(PGModel, self).get_variables(
            include_submodules=include_submodules,
            include_nontrainable=include_nontrainable
        )

        if self.baseline_mode is not None and (include_submodules or self.baseline_optimizer is None):
            baseline_variables = self.baseline.get_variables(include_nontrainable=include_nontrainable)
            model_variables += baseline_variables

            if include_nontrainable and self.baseline_optimizer is not None:
                baseline_optimizer_variables = self.baseline_optimizer.get_variables()
                # For some reason, some optimizer variables are only registered in the model.
                for variable in baseline_optimizer_variables:
                    if variable in model_variables:
                        model_variables.remove(variable)
                model_variables += baseline_optimizer_variables

        return model_variables

    def get_components(self):
        if self.baseline is None:
            return super(PGModel, self).get_components()
        else:
            result = dict(super(PGModel, self).get_components())
            result[PGModel.COMPONENT_BASELINE] = self.baseline
            return result
