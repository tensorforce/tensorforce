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
        states,
        actions,
        scope,
        device,
        saver,
        summarizer,
        execution,
        batching_capacity,
        action_values
    ):
        self.action_values = action_values

        super(ConstantModel, self).__init__(
            states=states,
            actions=actions,
            scope=scope,
            device=device,
            saver=saver,
            summarizer=summarizer,
            execution=execution,
            batching_capacity=batching_capacity,
            variable_noise=None,
            states_preprocessing=None,
            actions_exploration=None,
            reward_preprocessing=None
        )

    def tf_actions_and_internals(self, states, internals, deterministic):
        assert len(internals) == 0

        actions = dict()
        for name in sorted(self.actions_spec):
            shape = (tf.shape(input=states[next(iter(sorted(states)))])[0],) + self.actions_spec[name]['shape']
            actions[name] = tf.fill(dims=shape, value=self.action_values[name])

        return actions, dict()

    def tf_observe_timestep(self, states, internals, actions, terminal, reward):
        return tf.no_op()
