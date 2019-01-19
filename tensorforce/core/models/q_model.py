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

from tensorforce import util
from tensorforce.core import distribution_modules, network_modules, optimizer_modules, \
    parameter_modules
from tensorforce.core.models import DistributionModel


class QModel(DistributionModel):
    """
    Q-value model.
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
        # QModel
        target_sync_frequency, target_update_weight, double_q_model, huber_loss
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
            entropy_regularization=entropy_regularization, requires_deterministic=True
        )

        # Target network
        inputs_spec = OrderedDict()
        for name, spec in self.states_spec.items():
            inputs_spec[name] = dict(spec)
            inputs_spec[name]['batched'] = True
        self.target_network = self.add_module(
            name='target-network', module=network, modules=network_modules, is_trainable=False,
            is_subscope=True, inputs_spec=inputs_spec
        )
        embedding_size = self.target_network.get_output_spec()['shape'][0]

        # Target distributions
        self.target_distributions = OrderedDict()
        for name, action_spec in self.actions_spec.items():
            if action_spec['type'] == 'bool':
                default_module = 'bernoulli'
            elif action_spec['type'] == 'int':
                default_module = 'categorical'
            elif action_spec['type'] == 'float':
                if 'min_value' in action_spec:
                    default_module = 'beta'
                else:
                    default_module = 'gaussian'

            if distributions is None:
                module = None
            else:
                module = dict()
                if action_spec['type'] in distributions:
                    module.update(distributions[name])
                if name in distributions:
                    module.update(distributions[name])

            self.target_distributions[name] = self.add_module(
                name=(name + '-target-distribution'), module=module, modules=distribution_modules,
                default_module=default_module, is_trainable=False, action_spec=action_spec,
                embedding_size=embedding_size
            )

        # Target optimizer
        self.target_optimizer = self.add_module(
            name='target-optimizer', module='synchronization', modules=optimizer_modules,
            sync_frequency=target_sync_frequency, update_weight=target_update_weight
        )

        # Double Q-model
        self.double_q_model = double_q_model

        # Huber loss
        huber_loss = 0.0 if huber_loss is None else huber_loss
        self.huber_loss = self.add_module(
            name='huber-loss', module=huber_loss, modules=parameter_modules, dtype='float'
        )

    def as_local_model(self):
        super().as_local_model()
        self.target_optimizer_spec = dict(
            type='global_optimizer',
            optimizer=self.target_optimizer_spec
        )

    def tf_q_value(self, embedding, distr_params, action, name):
        # Mainly for NAF.
        return self.distributions[name].state_action_value(
            distr_params=distr_params, action=action
        )

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
        discount = self.discount.value()
        next_q_value = tf.where(condition=terminal, x=zeros, y=(discount * next_q_value))

        return reward + next_q_value - q_value  # tf.stop_gradient(q_target)

    def tf_loss_per_instance(
        self, states, internals, actions, terminal, reward, next_states, next_internals,
        reference=None
    ):
        embedding = self.network.apply(x=states, internals=internals)

        # fix
        if self.double_q_model:
            next_embedding = self.network.apply(x=next_states, internals=next_internals)

        # Both networks can use the same internals, could that be a problem?
        # Otherwise need to handle internals indices correctly everywhere
        target_embedding = self.target_network.apply(x=next_states, internals=next_internals)

        deltas = list()
        for name, distribution in self.distributions.items():
            target_distribution = self.target_distributions[name]

            distr_params = distribution.parametrize(x=embedding)
            target_distr_params = target_distribution.parametrize(x=target_embedding)

            q_value = self.tf_q_value(
                embedding=embedding, distr_params=distr_params, action=actions[name], name=name
            )

            if self.double_q_model:
                # fix
                next_distr_params = distribution.parametrize(x=next_embedding)
                action_taken = distribution.sample(
                    distr_params=next_distr_params, deterministic=True
                )
            else:
                action_taken = target_distribution.sample(
                    distr_params=target_distr_params, deterministic=True
                )

            next_q_value = target_distribution.state_action_value(
                distr_params=target_distr_params, action=action_taken
            )

            delta = self.q_delta(
                q_value=q_value, next_q_value=next_q_value, terminal=terminal, reward=reward
            )

            collapsed_size = util.product(xs=util.shape(delta)[1:])
            delta = tf.reshape(tensor=delta, shape=(-1, collapsed_size))

            deltas.append(delta)

        # Surrogate loss as the mean squared error between actual observed rewards and expected rewards
        loss_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=deltas, axis=1), axis=1)

        # Optional Huber loss
        huber_loss = self.huber_loss.value()

        def no_huber_loss():
            return tf.square(x=loss_per_instance)

        def apply_huber_loss():
            return tf.where(
                condition=(tf.abs(x=loss_per_instance) <= huber_loss),
                x=(0.5 * tf.square(x=loss_per_instance)),
                y=(huber_loss * (tf.abs(x=loss_per_instance) - 0.5 * huber_loss))
            )

        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        skip_huber_loss = tf.math.equal(x=huber_loss, y=zero)
        return self.cond(pred=skip_huber_loss, true_fn=no_huber_loss, false_fn=apply_huber_loss)

    def target_optimizer_arguments(self):
        """
        Returns the target optimizer arguments including the time, the list of variables to  
        optimize, and various functions which the optimizer might require to perform an update  
        step.

        Returns:
            Target optimizer arguments as dict.
        """
        variables = self.target_network.get_variables() + [
            variable for distribution in self.target_distributions.values()
            for variable in distribution.get_variables()
        ]
        source_variables = self.network.get_variables() + [
            variable for distribution in self.distributions.values()
            for variable in distribution.get_variables()
        ]
        arguments = dict(
            time=self.global_timestep, variables=variables, source_variables=source_variables
        )
        if self.global_model is not None:
            arguments['global_variables'] = self.global_model.target_network.get_variables() + [
                variable for distribution in self.global_model.target_distributions.values()
                for variable in distribution.get_variables()
            ]
        return arguments

    def tf_optimization(self, states, internals, actions, terminal, reward, next_states=None, next_internals=None):
        optimization = super().tf_optimization(
            states=states, internals=internals, actions=actions, terminal=terminal, reward=reward,
            next_states=next_states, next_internals=next_internals
        )

        arguments = self.target_optimizer_arguments()
        target_optimization = self.target_optimizer.minimize(**arguments)

        return tf.group(optimization, target_optimization)

    # # TEMP: Random sampling fix
    # def update(self, states, internals, actions, terminal, reward, return_loss_per_instance=False):
    #     fetches = [self.optimization]

    #     # Optionally fetch loss per instance
    #     if return_loss_per_instance:
    #         fetches.append(self.loss_per_instance)

    #     terminal = np.asarray(terminal)
    #     batched = (terminal.ndim == 1)
    #     if batched:
    #         # TEMP: Random sampling fix
    #         if self.random_sampling_fix:
    #             feed_dict = {state_input: states[name][0] for name, state_input in self.states_input)}
    #             feed_dict.update({state_input: states[name][1] for name, state_input in self.next_states_input)})
    #         else:
    #             feed_dict = {state_input: states[name] for name, state_input in self.states_input)}
    #         feed_dict.update(
    #             {internal_input: internals[n]
    #                 for n, internal_input in enumerate(self.internals_input)}
    #         )
    #         feed_dict.update(
    #             {action_input: actions[name]
    #                 for name, action_input in self.actions_input)}
    #         )
    #         feed_dict[self.terminal_input] = terminal
    #         feed_dict[self.reward_input] = reward
    #     else:
    #         # TEMP: Random sampling fix
    #         if self.random_sampling_fix:
    #             raise TensorForceError("Unbatched version not covered by fix.")
    #         else:
    #             feed_dict = {state_input: (states[name],) for name, state_input in self.states_input)}
    #         feed_dict.update(
    #             {internal_input: (internals[n],)
    #                 for n, internal_input in enumerate(self.internals_input)}
    #         )
    #         feed_dict.update(
    #             {action_input: (actions[name],)
    #                 for name, action_input in self.actions_input)}
    #         )
    #         feed_dict[self.terminal_input] = (terminal,)
    #         feed_dict[self.reward_input] = (reward,)

    #     feed_dict[self.deterministic_input] = True
    #     feed_dict[self.update_input] = True

    #     fetched = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

    #     if return_loss_per_instance:
    #         return fetched[1]
