# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

from tensorforce.core import parameter_modules, TensorSpec, tf_function, tf_util
from tensorforce.core.objectives import Objective


class Value(Objective):
    """
    Value approximation objective, which minimizes the L2-distance between the state-(action-)value
    estimate and the target reward value
    (specification key: `value`, `state_value`, `action_value`).

    Args:
        value ("state" | "action"): Whether to approximate the state- or state-action-value
            (<span style="color:#C00000"><b>required</b></span>).
        huber_loss (parameter, float > 0.0): Huber loss threshold
            (<span style="color:#00C000"><b>default</b></span>: no huber loss).
        early_reduce (bool): Whether to compute objective for aggregated value instead of value per
            action (<span style="color:#00C000"><b>default</b></span>: true).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        reward_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, value, huber_loss=None, early_reduce=True, name=None, states_spec=None,
        internals_spec=None, auxiliaries_spec=None, actions_spec=None, reward_spec=None
    ):
        super().__init__(
            name=name, states_spec=states_spec, internals_spec=internals_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec, reward_spec=reward_spec
        )

        assert value in ('state', 'action')
        self.value = value

        if huber_loss is None:
            self.huber_loss = None
        else:
            self.huber_loss = self.submodule(
                name='huber_loss', module=huber_loss, modules=parameter_modules, dtype='float',
                min_value=0.0
            )

        self.early_reduce = early_reduce

    def required_policy_fns(self):
        if self.value == 'state':
            return ('state_value',)
        elif self.value == 'action':
            return ('action_value',)

    def reference_spec(self):
        # if self.early_reduce:
        return TensorSpec(type='float', shape=())

        # else:
        #     return TensorSpec(
        #         type='float', shape=(sum(spec.size for spec in self.actions_spec.values()),)
        #     )

    @tf_function(num_args=5)
    def reference(self, *, states, horizons, internals, auxiliaries, actions, policy):
        # if self.value == 'state':
        #     if self.early_reduce:
        #         value = policy.state_value(
        #             states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        #         )
        #     else:
        #         value = policy.state_values(
        #             states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
        #         )
        #         value = tf.concat(values=tuple(value.values()), axis=1)

        # elif self.value == 'action':
        #     if self.early_reduce:
        #         value = policy.action_value(
        #             states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
        #             actions=actions
        #         )
        #     else:
        #         value = policy.action_values(
        #             states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
        #             actions=actions
        #         )
        #         value = tf.concat(values=tuple(value.values()), axis=1)

        return tf_util.zeros(shape=(tf.shape(input=actions.value())[0],), dtype='float')

    @tf_function(num_args=7)
    def loss(
        self, *, states, horizons, internals, auxiliaries, actions, reward, reference, policy,
        baseline=None
    ):
        # value = self.reference(
        #     states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
        #     actions=actions, policy=policy
        # )

        # reference = tf.stop_gradient(input=reference)

        if self.value == 'state':
            if self.early_reduce:
                value = policy.state_value(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
                )
            else:
                value = policy.state_values(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries
                )
                value = tf.concat(values=tuple(value.values()), axis=1)

        elif self.value == 'action':
            if self.early_reduce:
                value = policy.action_value(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    actions=actions
                )
            else:
                value = policy.action_values(
                    states=states, horizons=horizons, internals=internals, auxiliaries=auxiliaries,
                    actions=actions
                )
                value = tf.concat(values=tuple(value.values()), axis=1)

        if not self.early_reduce:
            reward = tf.expand_dims(input=reward, axis=1)

        difference = value - reward

        half = tf_util.constant(value=0.5, dtype='float')

        if self.huber_loss is None:
            loss = half * tf.math.square(x=difference)

        else:
            huber_loss = self.huber_loss.value()
            inside_huber_bounds = tf.math.less_equal(x=tf.math.abs(x=difference), y=huber_loss)
            quadratic = half * tf.math.square(x=difference)
            linear = huber_loss * (tf.math.abs(x=difference) - half * huber_loss)
            loss = tf.where(condition=inside_huber_bounds, x=quadratic, y=linear)

        if not self.early_reduce:
            loss = tf.math.reduce_sum(input_tensor=loss, axis=1)

        return loss
