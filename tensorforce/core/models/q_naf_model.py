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

from tensorforce import TensorforceError, util
from tensorforce.core import layer_modules
from tensorforce.core.models import QModel


class QNAFModel(QModel):
    """
    Implements normalized advantage functions (NAF), somtimes also called
    continuous Q-learning.
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
        if any(spec['type'] != 'float' or 'min_value' in spec or 'max_value' in spec for name, spec in actions.items()):
            raise TensorforceError("Only unconstrained float actions valid for NAFModel.")

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
            entropy_regularization=entropy_regularization,
            # QModel
            target_sync_frequency=target_sync_frequency, target_update_weight=target_update_weight,
            double_q_model=double_q_model, huber_loss=huber_loss
        )

        self.state_values = OrderedDict()
        self.l_entries = OrderedDict()
        embedding_size = self.network.get_output_spec()['shape'][0]
        input_spec = dict(type='float', shape=(embedding_size,))
        for name, action_spec in self.actions_spec.items():
            action_size = util.product(xs=action_spec['shape'])
            self.state_values[name] = self.add_module(
                name=(name + '-state-value'), module='linear', modules=layer_modules,
                size=action_size, input_spec=input_spec
            )
            self.l_entries[name] = self.add_module(
                name=(name + '-l-entries'), module='linear', modules=layer_modules,
                size=action_size, input_spec=input_spec
            )

    def tf_q_value(self, embedding, distr_params, action, name):
        num_action = util.product(xs=self.actions_spec[name]['shape'])

        mean, stddev, _ = distr_params
        flat_mean = tf.reshape(tensor=mean, shape=(-1, num_action))
        flat_stddev = tf.reshape(tensor=stddev, shape=(-1, num_action))

        # Advantage computation
        # Network outputs entries of lower triangular matrix L
        if self.l_entries[name] is None:
            l_matrix = flat_stddev
            l_matrix = tf.exp(l_matrix)
        else:
            l_matrix = tf.linalg.diag(diagonal=flat_stddev)

            l_entries = self.l_entries[name].apply(x=embedding)
            l_entries = tf.exp(l_entries)
            offset = 0
            columns = list()
            for zeros, size in enumerate(range(num_action - 1, -1, -1), 1):
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

    def tf_loss_per_instance(
        self, states, internals, actions, terminal, reward, next_states, next_internals,
        reference=None
    ):
        # Really state value instead of q value?
        # Michael: doubling this function because NAF needs V'(s) not Q'(s), see comment below
        embedding = self.network.apply(x=states, internals=internals)

        # Both networks can use the same internals, could that be a problem?
        # Otherwise need to handle internals indices correctly everywhere
        target_embedding = self.target_network.apply(x=next_states, internals=next_internals)

        deltas = list()
        for name in sorted(self.distributions):
            distribution = self.distributions[name]
            target_distribution = self.target_distributions[name]

            distr_params = distribution.parametrize(x=embedding)
            target_distr_params = target_distribution.parametrize(x=target_embedding)

            q_value = self.tf_q_value(
                embedding=embedding, distr_params=distr_params, action=actions[name], name=name
            )

            # Notice, this is V', not Q' because NAF outputs V(s) separately
            next_state_value = target_distribution.state_value(distr_params=target_distr_params)

            delta = self.tf_q_delta(
                q_value=q_value, next_q_value=next_state_value, terminal=terminal, reward=reward
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
