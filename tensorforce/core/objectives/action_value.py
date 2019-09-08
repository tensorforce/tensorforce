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

import tensorflow as tf

from tensorforce import util
from tensorforce.core import parameter_modules
from tensorforce.core.objectives import Objective


class ActionValue(Objective):
    """
    State-action-value / Q-value objective, which minimizes the L2-distance between the
    state-action-value estimate and target reward value (specification key: `action_value`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        huber_loss (parameter, float > 0.0): Huber loss threshold
            (<span style="color:#00C000"><b>default</b></span>: no huber loss).
        mean_over_actions (bool): Whether to compute objective for mean of state-action-values
            instead of per state-action-value
            (<span style="color:#00C000"><b>default</b></span>: false).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, huber_loss=0.0, mean_over_actions=False, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels)

        huber_loss = 0.0 if huber_loss is None else huber_loss
        self.huber_loss = self.add_module(
            name='huber-loss', module=huber_loss, modules=parameter_modules, dtype='float'
        )

        self.mean_over_actions = mean_over_actions

    def tf_loss_per_instance(self, policy, states, internals, auxiliaries, actions, reward):
        if not self.mean_over_actions:
            reward = tf.expand_dims(input=reward, axis=1)

        actions_value = policy.actions_value(
            states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
            mean=self.mean_over_actions
        )
        difference = actions_value - reward

        zero = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
        half = tf.constant(value=0.5, dtype=util.tf_dtype(dtype='float'))

        huber_loss = self.huber_loss.value()
        skip_huber_loss = tf.math.equal(x=huber_loss, y=zero)

        def no_huber_loss():
            return half * tf.square(x=difference)

        def apply_huber_loss():
            inside_huber_bounds = tf.math.less_equal(x=tf.abs(x=difference), y=huber_loss)
            quadratic = half * tf.square(x=difference)
            linear = huber_loss * (tf.abs(x=difference) - half * huber_loss)
            return tf.where(condition=inside_huber_bounds, x=quadratic, y=linear)

        loss = self.cond(pred=skip_huber_loss, true_fn=no_huber_loss, false_fn=apply_huber_loss)

        if not self.mean_over_actions:
            loss = tf.math.reduce_mean(input_tensor=loss, axis=1)

        return loss
