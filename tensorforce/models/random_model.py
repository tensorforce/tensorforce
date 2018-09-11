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

from tensorforce import util
from tensorforce.models import Model


class RandomModel(Model):
    """
    Utility class to return random actions of a desired shape and with given bounds.
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
        batching_capacity
    ):
        super(RandomModel, self).__init__(
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
            action = self.actions_spec[name]
            shape = (tf.shape(input=states[next(iter(sorted(states)))])[0],) + action['shape']

            if action['type'] == 'bool':
                actions[name] = (tf.random_uniform(shape=shape) < 0.5)

            elif action['type'] == 'int':
                actions[name] = tf.random_uniform(shape=shape, maxval=action['num_actions'], dtype=util.tf_dtype('int'))

            elif action['type'] == 'float':
                if 'min_value' in action:
                    actions[name] = tf.random_uniform(
                        shape=shape,
                        minval=action['min_value'],
                        maxval=action['max_value']
                    )

                else:
                    actions[name] = tf.random_normal(shape=shape)

        return actions, dict()

    def tf_observe_timestep(self, states, internals, actions, terminal, reward):
        return tf.no_op()
