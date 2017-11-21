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

from six.moves import xrange

import tensorflow as tf

from tensorforce import util, TensorForceError
from tensorforce.models import QModel
from tensorforce.core.networks import Linear


class QNAFModel(QModel):

    def __init__(
        self,
        states_spec,
        actions_spec,
        network_spec,
        device,
        scope,
        saver_spec,
        summary_spec,
        distributed_spec,
        optimizer,
        discount,
        normalize_rewards,
        variable_noise,
        distributions_spec,
        entropy_regularization,
        target_sync_frequency,
        target_update_weight,
        double_q_model,
        huber_loss,
        # TEMP: Random sampling fix
        random_sampling_fix
    ):
        if any(action['type'] != 'float' or 'min_value' in action or 'max_value' in action for action in actions_spec.values()):
            raise TensorForceError("Only unconstrained float actions valid for NAFModel.")

        super(QNAFModel, self).__init__(
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
        super(QNAFModel, self).initialize(custom_getter)

        self.state_values = dict()
        self.l_entries = dict()
        for name, action in self.actions_spec.items():
            num_action = util.prod(action['shape'])
            self.state_values[name] = Linear(size=num_action, scope='state-value')
            self.l_entries[name] = Linear(size=(num_action * (num_action - 1) // 2), scope='l-entries')

    def tf_q_value(self, embedding, distr_params, action, name):
        num_action = util.prod(self.actions_spec[name]['shape'])

        mean, stddev, _ = distr_params
        flat_mean = tf.reshape(tensor=mean, shape=(-1, num_action))
        flat_stddev = tf.reshape(tensor=stddev, shape=(-1, num_action))

        # Advantage computation
        # Network outputs entries of lower triangular matrix L
        if self.l_entries[name] is None:
            l_matrix = flat_stddev

        else:
            l_matrix = tf.map_fn(fn=tf.diag, elems=flat_stddev)

            l_entries = self.l_entries[name].apply(x=embedding)

            offset = 0
            columns = list()
            for zeros, size in enumerate(xrange(num_action - 1, -1, -1), 1):
                column = tf.pad(tensor=l_entries[:, offset: offset + size], paddings=((0, 0), (zeros, 0)))
                columns.append(column)
                offset += size

            l_matrix += tf.stack(values=columns, axis=1)

        # P = LL^T
        p_matrix = tf.matmul(a=l_matrix, b=tf.transpose(a=l_matrix, perm=(0, 2, 1)))

        # A = -0.5 (a - mean)P(a - mean)
        flat_action = tf.reshape(tensor=action, shape=(-1, num_action))
        difference = flat_action - flat_mean
        advantage = tf.matmul(a=p_matrix, b=tf.expand_dims(input=difference, axis=2))
        advantage = tf.matmul(a=tf.expand_dims(input=difference, axis=1), b=advantage)
        advantage = tf.squeeze(input=(-advantage / 2.0), axis=2)

        # Q = A + V
        # State-value function
        state_value = self.state_values[name].apply(x=embedding)
        q_value = state_value + advantage

        return tf.reshape(tensor=q_value, shape=((-1,) + self.actions_spec[name]['shape']))

    def tf_regularization_losses(self, states, internals, update):
        losses = super(QNAFModel, self).tf_regularization_losses(
            states=states,
            internals=internals,
            update=update
        )

        for state_value in self.state_values.values():
            regularization_loss = state_value.regularization_loss()
            if regularization_loss is not None:
                if 'state-values' in losses:
                    losses['state-values'] += regularization_loss
                else:
                    losses['state-values'] = regularization_loss

        for l_entries in self.l_entries.values():
            regularization_loss = l_entries.regularization_loss()
            if regularization_loss is not None:
                if 'l-entries' in losses:
                    losses['l-entries'] += regularization_loss
                else:
                    losses['l-entries'] = regularization_loss

        return losses

    def get_variables(self, include_non_trainable=False):
        model_variables = super(QNAFModel, self).get_variables(include_non_trainable=include_non_trainable)

        state_values_variables = [
            variable for name in sorted(self.state_values)
            for variable in self.state_values[name].get_variables()
        ]

        l_entries_variables = [
            variable for name in sorted(self.l_entries)
            for variable in self.l_entries[name].get_variables()
        ]

        return model_variables + state_values_variables + l_entries_variables
