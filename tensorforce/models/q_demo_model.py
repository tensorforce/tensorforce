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
from tensorforce import util, TensorForceError
from tensorforce.models import QModel


class QDemoModel(QModel):
    """
    Model for deep Q-learning from demonstration. Principal structure similar to double deep Q-networks but uses additional loss terms for demo data.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        device,
        scope,
        saver_spec,
        summary_spec,
        distributed_spec,
        optimizer,
        discount,
        normalize_rewards,
        variable_noise,
        network_spec,
        distributions_spec,
        entropy_regularization,
        target_sync_frequency,
        target_update_weight,
        double_q_model,
        huber_loss,
        # TEMP: Random sampling fix
        random_sampling_fix,
        expert_margin,
        supervised_weight
    ):
        if any(action['type'] not in ('bool', 'int') for action in actions_spec.values()):
            raise TensorForceError("Invalid action type, only 'bool' and 'int' are valid!")

        self.expert_margin = expert_margin
        self.supervised_weight = supervised_weight

        super(QDemoModel, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=network_spec,
            device=device,
            scope=scope,
            saver_spec=saver_spec,
            summary_spec=summary_spec,
            distributed_spec=distributed_spec,
            optimizer=optimizer,
            discount=discount,
            normalize_rewards=normalize_rewards,
            variable_noise=variable_noise,
            distributions_spec=distributions_spec,
            entropy_regularization=entropy_regularization,
            target_sync_frequency=target_sync_frequency,
            target_update_weight=target_update_weight,
            double_q_model=double_q_model,
            huber_loss=huber_loss,
            # TEMP: Random sampling fix
            random_sampling_fix=random_sampling_fix
        )

    def initialize(self, custom_getter):
        super(QDemoModel, self).initialize(custom_getter=custom_getter)

        # Demonstration loss
        self.fn_demo_loss = tf.make_template(
            name_='demo-loss',
            func_=self.tf_demo_loss,
            custom_getter_=custom_getter
        )

        # Demonstration optimization
        self.fn_demo_optimization = tf.make_template(
            name_='demo-optimization',
            func_=self.tf_demo_optimization,
            custom_getter_=custom_getter
        )

    def create_output_operations(self, states, internals, actions, terminal, reward, update, deterministic):
        super(QDemoModel, self).create_output_operations(
            states=states,
            internals=internals,
            actions=actions,
            reward=reward,
            terminal=terminal,
            update=update,
            deterministic=deterministic
        )

        self.demo_optimization = self.fn_optimization(
            states=states,
            internals=internals,
            actions=actions,
            reward=reward,
            terminal=terminal,
            update=update
        )

    def tf_demo_loss(self, states, actions, terminal, reward, internals, update):
        embedding = self.network.apply(x=states, internals=internals, update=update)
        deltas = list()

        for name, distribution in self.distributions.items():
            distr_params = distribution.parameters(x=embedding)
            state_action_values = distribution.state_action_values(distr_params=distr_params)

            # Create the supervised margin loss
            # Zero for the action taken, one for all other actions, now multiply by expert margin
            if self.actions_spec[name]['type'] == 'bool':
                num_actions = 2
            else:
                num_actions = self.actions_spec[name]['num_actions']
            one_hot = tf.one_hot(indices=actions[name], depth=num_actions)
            ones = tf.ones_like(tensor=one_hot, dtype=tf.float32)
            inverted_one_hot = ones - one_hot

            # max_a([Q(s,a) + l(s,a_E,a)], l(s,a_E, a) is 0 for expert action and margin value for others
            expert_margin = distr_params + inverted_one_hot * self.expert_margin

            # J_E(Q) = max_a([Q(s,a) + l(s,a_E,a)] - Q(s,a_E)
            supervised_selector = tf.reduce_max(input_tensor=expert_margin, axis=-1)
            delta = supervised_selector - state_action_values
            delta = tf.reshape(tensor=delta, shape=(-1, util.prod(self.actions_spec[name]['shape'])))
            deltas.append(delta)

        loss_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=deltas, axis=1), axis=1)
        loss_per_instance = tf.square(x=loss_per_instance)
        return tf.reduce_mean(input_tensor=loss_per_instance, axis=0)

    def tf_demo_optimization(self, states, internals, actions, terminal, reward, update):

        def fn_loss():
            # Combining q-loss with demonstration loss
            q_model_loss = self.fn_loss(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                update=update
            )
            demo_loss = self.fn_demo_loss(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                update=update
            )
            return q_model_loss + self.supervised_weight * demo_loss

        demo_optimization = self.optimizer.minimize(
            time=self.timestep,
            variables=self.get_variables(),
            fn_loss=fn_loss
        )

        target_optimization = self.target_optimizer.minimize(
            time=self.timestep,
            variables=self.target_network.get_variables(),
            source_variables=self.network.get_variables()
        )

        return tf.group(demo_optimization, target_optimization)

    def demonstration_update(self, batch):
        fetches = self.demo_optimization

        feed_dict = {state_input: batch['states'][name] for name, state_input in self.state_inputs.items()}
        feed_dict.update(
            {internal_input: batch['internals'][n]
                for n, internal_input in enumerate(self.internal_inputs)}
        )
        feed_dict.update(
            {action_input: batch['actions'][name]
                for name, action_input in self.action_inputs.items()}
        )
        feed_dict[self.terminal_input] = batch['terminal']
        feed_dict[self.reward_input] = batch['reward']

        # TODO: summaries? distributed?

        self.session.run(fetches=fetches, feed_dict=feed_dict)
