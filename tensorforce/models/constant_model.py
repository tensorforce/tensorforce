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

from tensorforce.models import Model


class ConstantModel(Model):
    """
    Utility class to return constant actions of a desired shape and with given bounds.
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
        action_values
    ):
        self.action_values = action_values

        super(ConstantModel, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            device=device,
            scope=scope,
            saver_spec=saver_spec,
            summary_spec=summary_spec,
            distributed_spec=distributed_spec,
            optimizer=optimizer,
            discount=discount,
            normalize_rewards=normalize_rewards,
            variable_noise=variable_noise
        )

    def tf_actions_and_internals(self, states, internals, update, deterministic):
        actions = dict()
        for name, action in self.actions_spec.items():
            shape = (tf.shape(input=next(iter(states.values())))[0],) + action['shape']
            actions[name] = tf.fill(dims=shape, value=self.action_values[name])

        return actions, internals

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward, update):
        # Nothing to be done here, loss is 0.
        return tf.zeros_like(tensor=reward)
