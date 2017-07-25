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
Model for deep-q learning from demonstration. Principal structure similar to double deep-q-networks
but uses additional loss terms for demo data.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorforce.models import DQNModel


class DQFDModel(DQNModel):

    default_config = dict(
        double_dqn=True,
        supervised_weight=1.0,
        expert_margin=0.8
    )

    def __init__(self, config):
        config.default(DQFDModel.default_config)
        super(DQFDModel, self).__init__(config)

    def create_tf_operations(self, config):
        """Create training graph. For DQFD, we build the double-dqn training graph and
        modify the double_q_loss function according to eq. 5
        
        Args:
            config: Config dict.

        Returns:

        """
        super(DQFDModel, self).create_tf_operations(config)

        with tf.name_scope('supervised-update'):
            deltas = list()
            for action in self.action:
                # Create the supervised margin loss
                mask = tf.ones_like(self.actions_one_hot[action], dtype=tf.float32)

                # Zero for the action taken, one for all other actions, now multiply by expert margin
                inverted_one_hot = mask - self.actions_one_hot[action]

                # max_a([Q(s,a) + l(s,a_E,a)], l(s,a_E, a) is 0 for expert action and margin value for others
                expert_margin = self.training_output[action][:-1] + inverted_one_hot * config.expert_margin

                supervised_selector = tf.reduce_max(input_tensor=expert_margin, axis=1)

                # J_E(Q) = max_a([Q(s,a) + l(s,a_E,a)] - Q(s,a_E)
                delta = supervised_selector - self.q_values[action]

                ds_list = [delta]
                for _ in range(len(config.actions[action].shape)):
                    ds_list = [d for ds in ds_list for d in tf.unstack(value=ds, axis=1)]
                deltas.extend(ds_list)

            delta = tf.add_n(inputs=deltas) / len(deltas)
            supervised_loss_per_instance = tf.square(delta)
            supervised_loss = tf.reduce_mean(input_tensor=supervised_loss_per_instance)

            # Combining double q loss with supervised loss
            dqfd_loss = self.dqn_loss + supervised_loss * config.supervised_weight
            self.dqfd_optimize = self.optimizer.minimize(dqfd_loss)

    def demonstration_update(self, batch=None):
        """Computes the demonstration update.

        Args:
            batch: A batch of demo data.

        Returns:

        """

        fetches = self.dqfd_optimize

        feed_dict = {state: batch['states'][name] for name, state in self.state.items()}
        feed_dict.update({action: batch['actions'][name] for name, action in self.action.items()})
        feed_dict[self.reward] = batch['rewards']
        feed_dict[self.terminal] = batch['terminals']
        feed_dict.update({internal: batch['internals'][n] for n, internal in enumerate(self.internal_inputs)})

        self.session.run(fetches=fetches, feed_dict=feed_dict)
