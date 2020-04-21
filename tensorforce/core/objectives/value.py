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

from tensorforce.core import parameter_modules, tf_function, tf_util
from tensorforce.core.objectives import Objective


class Value(Objective):
    """
    Value approximation objective, which minimizes the L2-distance between the state-(action-)value
    estimate and the target reward value (specification key: `value`).

    Args:
        value ("state" | "action"): Whether to approximate the state- or state-action-value
            (<span style="color:#00C000"><b>default</b></span>: "state").
        huber_loss (parameter, float >= 0.0): Huber loss threshold
            (<span style="color:#00C000"><b>default</b></span>: no huber loss).
        early_reduce (bool): Whether to compute objective for reduced values instead of value per
            action (<span style="color:#00C000"><b>default</b></span>: true).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        reward_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, value='state', huber_loss=0.0, early_reduce=True, summary_labels=None, name=None,
        states_spec=None, internals_spec=None, auxiliaries_spec=None, actions_spec=None,
        reward_spec=None
    ):
        super().__init__(
            summary_labels=summary_labels, name=name, states_spec=states_spec,
            internals_spec=internals_spec, auxiliaries_spec=auxiliaries_spec,
            actions_spec=actions_spec, reward_spec=reward_spec
        )

        assert value in ('state', 'action')
        self.value = value

        huber_loss = 0.0 if huber_loss is None else huber_loss
        self.huber_loss = self.add_module(
            name='huber_loss', module=huber_loss, modules=parameter_modules, dtype='float',
            min_value=0.0
        )

        self.early_reduce = early_reduce

    @tf_function(num_args=6)
    def loss(self, *, states, horizons, internals, auxiliaries, actions, reward, policy):
        if not self.early_reduce:
            reward = tf.expand_dims(input=reward, axis=1)

        if self.value == 'state':
            value = policy.states_value(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                reduced=self.early_reduce, return_per_action=False
            )
        elif self.value == 'action':
            value = policy.actions_value(
                states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                actions=actions, reduced=self.early_reduce, return_per_action=False
            )

        difference = value - reward

        zero = tf_util.constant(value=0.0, dtype='float')
        half = tf_util.constant(value=0.5, dtype='float')

        huber_loss = self.huber_loss.value()
        skip_huber_loss = tf.math.equal(x=huber_loss, y=zero)

        def no_huber_loss():
            return half * tf.math.square(x=difference)

        def apply_huber_loss():
            inside_huber_bounds = tf.math.less_equal(x=tf.math.abs(x=difference), y=huber_loss)
            quadratic = half * tf.math.square(x=difference)
            linear = huber_loss * (tf.math.abs(x=difference) - half * huber_loss)
            return tf.where(condition=inside_huber_bounds, x=quadratic, y=linear)

        loss = tf.cond(pred=skip_huber_loss, true_fn=no_huber_loss, false_fn=apply_huber_loss)

        if not self.early_reduce:
            loss = tf.math.reduce_mean(input_tensor=loss, axis=1)

        return loss
