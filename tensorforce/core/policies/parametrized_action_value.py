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

from tensorforce.core import layer_modules, TensorDict, TensorsSpec, tf_function, tf_util
from tensorforce.core.policies import ActionValue, ParametrizedPolicy


class ParametrizedActionValue(ActionValue, ParametrizedPolicy):
    """
    Policy which parametrizes an action-value function, conditioned on the output of a neural
    network processing the input state (specification key: `parametrized_action_value`).

    Args:
        network ('auto' | specification): Policy network configuration, see
            [networks](../modules/networks.html)
            (<span style="color:#00C000"><b>default</b></span>: 'auto', automatically configured
            network).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        states_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        auxiliaries_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        internals_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
        actions_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    # Network first
    def __init__(
        self, network='auto', *, device=None, l2_regularization=None, name=None, states_spec=None,
        auxiliaries_spec=None, internals_spec=None, actions_spec=None
    ):
        super().__init__(
            device=device, l2_regularization=l2_regularization, name=name, states_spec=states_spec,
            auxiliaries_spec=auxiliaries_spec, actions_spec=actions_spec
        )

        inputs_spec = TensorsSpec()
        if self.states_spec.is_singleton():
            inputs_spec['states'] = self.states_spec.singleton()
        else:
            inputs_spec['states'] = self.states_spec
        if self.actions_spec.is_singleton():
            inputs_spec['actions'] = self.actions_spec.singleton()
        else:
            inputs_spec['actions'] = self.actions_spec
        ParametrizedPolicy.__init__(self=self, network=network, inputs_spec=inputs_spec)
        output_spec = self.network.output_spec()

        # Action value
        self.value = self.submodule(
            name='value', module='linear', modules=layer_modules, size=0, input_spec=output_spec
        )

    def get_architecture(self):
        return 'Network:  {}\nAction-value:  {}'.format(
            self.network.get_architecture().replace('\n', '\n    '),
            self.value.get_architecture().replace('\n', '\n    ')
        )

    @tf_function(num_args=5)
    def next_internals(self, *, states, horizons, internals, actions, deterministic, independent):
        inputs = TensorDict()
        if self.states_spec.is_singleton():
            inputs['states'] = states.singleton()
        else:
            inputs['states'] = states
        if self.actions_spec.is_singleton():
            inputs['actions'] = actions.singleton()
        else:
            inputs['actions'] = actions

        return super().next_internals(
            states=inputs, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=independent
        )

    @tf_function(num_args=5)
    def action_value(self, *, states, horizons, internals, auxiliaries, actions):
        inputs = TensorDict()
        if self.states_spec.is_singleton():
            inputs['states'] = states.singleton()
        else:
            inputs['states'] = states
        if self.actions_spec.is_singleton():
            inputs['actions'] = actions.singleton()
        else:
            inputs['actions'] = actions

        deterministic = tf_util.constant(value=True, dtype='bool')
        embedding, _ = self.network.apply(
            x=inputs, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=True
        )

        return self.value.apply(x=embedding)
