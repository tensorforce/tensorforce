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
from tensorforce.core.memories import Replay
from tensorforce.models import QModel


class QDemoModel(QModel):
    """
    Model for deep Q-learning from demonstration. Principal structure similar to double
    deep Q-networks but uses additional loss terms for demo data.
    """

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
        target_sync_frequency,
        target_update_weight,
        double_q_model,
        huber_loss,
        expert_margin,
        supervised_weight,
        demo_memory_capacity,
        demo_batch_size
    ):
        if any(actions[name]['type'] not in ('bool', 'int') for name in sorted(actions)):
            raise TensorForceError("Invalid action type, only 'bool' and 'int' are valid!")

        self.expert_margin = expert_margin
        self.supervised_weight = supervised_weight
        self.demo_memory_capacity = demo_memory_capacity
        self.demo_batch_size = demo_batch_size

        self.demo_memory = None
        self.fn_import_demo_experience = None
        self.fn_demo_loss = None
        self.fn_combined_loss = None
        self.fn_demo_optimization = None

        super(QDemoModel, self).__init__(
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
            target_sync_frequency=target_sync_frequency,
            target_update_weight=target_update_weight,
            double_q_model=double_q_model,
            huber_loss=huber_loss
        )

    def setup_components_and_tf_funcs(self, custom_getter=None):
        """
        Constructs the extra Replay memory.
        """
        custom_getter = super(QDemoModel, self).setup_components_and_tf_funcs(custom_getter)

        self.demo_memory = Replay(
            states=self.states_spec,
            internals=self.internals_spec,
            actions=self.actions_spec,
            include_next_states=True,
            capacity=self.demo_memory_capacity,
            scope='demo-replay',
            summary_labels=self.summary_labels
        )

        # Import demonstration optimization.
        self.fn_import_demo_experience = tf.make_template(
            name_='import-demo-experience',
            func_=self.tf_import_demo_experience,
            custom_getter_=custom_getter
        )

        # Demonstration loss.
        self.fn_demo_loss = tf.make_template(
            name_='demo-loss',
            func_=self.tf_demo_loss,
            custom_getter_=custom_getter
        )

        # Combined loss.
        self.fn_combined_loss = tf.make_template(
            name_='combined-loss',
            func_=self.tf_combined_loss,
            custom_getter_=custom_getter
        )

        # Demonstration optimization.
        self.fn_demo_optimization = tf.make_template(
            name_='demo-optimization',
            func_=self.tf_demo_optimization,
            custom_getter_=custom_getter
        )

        return custom_getter

    def tf_initialize(self):
        super(QDemoModel, self).tf_initialize()
        self.demo_memory.initialize()

    def tf_import_demo_experience(self, states, internals, actions, terminal, reward):
        """
        Imports a single experience to memory.
        """
        return self.demo_memory.store(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

    def tf_demo_loss(self, states, actions, terminal, reward, internals, update, reference=None):
        """
        Extends the q-model loss via the dqfd large-margin loss.
        """
        embedding = self.network.apply(x=states, internals=internals, update=update)
        deltas = list()

        for name in sorted(actions):
            action = actions[name]
            distr_params = self.distributions[name].parameterize(x=embedding)
            state_action_value = self.distributions[name].state_action_value(distr_params=distr_params, action=action)

            # Create the supervised margin loss
            # Zero for the action taken, one for all other actions, now multiply by expert margin
            if self.actions_spec[name]['type'] == 'bool':
                num_actions = 2
                action = tf.cast(x=action, dtype=util.tf_dtype('int'))
            else:
                num_actions = self.actions_spec[name]['num_actions']

            one_hot = tf.one_hot(indices=action, depth=num_actions)
            ones = tf.ones_like(tensor=one_hot, dtype=tf.float32)
            inverted_one_hot = ones - one_hot

            # max_a([Q(s,a) + l(s,a_E,a)], l(s,a_E, a) is 0 for expert action and margin value for others
            state_action_values = self.distributions[name].state_action_value(distr_params=distr_params)
            state_action_values = state_action_values + inverted_one_hot * self.expert_margin
            supervised_selector = tf.reduce_max(input_tensor=state_action_values, axis=-1)

            # J_E(Q) = max_a([Q(s,a) + l(s,a_E,a)] - Q(s,a_E)
            delta = supervised_selector - state_action_value

            action_size = util.prod(self.actions_spec[name]['shape'])
            delta = tf.reshape(tensor=delta, shape=(-1, action_size))
            deltas.append(delta)

        loss_per_instance = tf.reduce_mean(input_tensor=tf.concat(values=deltas, axis=1), axis=1)
        loss_per_instance = tf.square(x=loss_per_instance)

        return tf.reduce_mean(input_tensor=loss_per_instance, axis=0)

    def tf_combined_loss(self, states, internals, actions, terminal, reward, next_states, next_internals, update, reference=None):
        """
        Combines Q-loss and demo loss.
        """
        q_model_loss = self.fn_loss(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            next_states=next_states,
            next_internals=next_internals,
            update=update,
            reference=reference
        )

        demo_loss = self.fn_demo_loss(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            update=update,
            reference=reference
        )

        return q_model_loss + self.supervised_weight * demo_loss

    def tf_demo_optimization(self, states, internals, actions, terminal, reward, next_states, next_internals):
        arguments = dict(
            time=self.global_timestep,
            variables=self.get_variables(),
            arguments=dict(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                next_states=next_states,
                next_internals=next_internals,
                update=tf.constant(value=True)
            ),
            fn_loss=self.fn_combined_loss
        )
        demo_optimization = self.optimizer.minimize(**arguments)

        arguments = self.target_optimizer_arguments()
        target_optimization = self.target_optimizer.minimize(**arguments)

        return tf.group(demo_optimization, target_optimization)

    def tf_optimization(self, states, internals, actions, terminal, reward, next_states=None, next_internals=None):
        optimization = super(QDemoModel, self).tf_optimization(
            states=states,
            internals=internals,
            actions=actions,
            reward=reward,
            terminal=terminal,
            next_states=next_states,
            next_internals=next_internals
        )

        demo_batch = self.demo_memory.retrieve_timesteps(n=self.demo_batch_size)
        demo_optimization = self.fn_demo_optimization(**demo_batch)

        return tf.group(optimization, demo_optimization)

    def create_operations(self, states, internals, actions, terminal, reward, deterministic, independent, index):
        # Import demo experience operation.
        self.import_demo_experience_output = self.fn_import_demo_experience(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

        # !!!
        super(QDemoModel, self).create_operations(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            deterministic=deterministic,
            independent=independent,
            index=index
        )

        # Demo optimization operation.
        demo_batch = self.demo_memory.retrieve_timesteps(n=self.demo_batch_size)
        self.demo_optimization_output = self.fn_demo_optimization(**demo_batch)

    def get_variables(self, include_submodules=False, include_nontrainable=False):
        """
        Returns the TensorFlow variables used by the model.

        Returns:
            List of variables.
        """
        model_variables = super(QDemoModel, self).get_variables(
            include_submodules=include_submodules,
            include_nontrainable=include_nontrainable
        )

        if include_nontrainable:
            demo_memory_variables = self.demo_memory.get_variables()
            model_variables += demo_memory_variables

        return model_variables

    def import_demo_experience(self, states, internals, actions, terminal, reward):
        """
        Stores demonstrations in the demo memory.
        """
        fetches = self.import_demo_experience_output

        feed_dict = self.get_feed_dict(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward
        )

        self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

    def demo_update(self):
        """
        Performs a demonstration update by calling the demo optimization operation.
        Note that the batch data does not have to be fetched from the demo memory as this is now part of
        the TensorFlow operation of the demo update.
        """
        fetches = self.demo_optimization_output

        self.monitored_session.run(fetches=fetches)
