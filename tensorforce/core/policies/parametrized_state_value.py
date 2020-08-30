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

from tensorforce import TensorforceError
from tensorforce.core import layer_modules, network_modules, tf_function, tf_util
from tensorforce.core.policies import StateValue


class ParametrizedStateValue(StateValue):
    """
    Policy which parametrizes a state-value function, conditioned on the output of a neural network
    processing the input state (specification key: `parametrized_state_value`).

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

        # Network
        self.network = self.submodule(
            name='network', module=network, modules=network_modules, inputs_spec=self.states_spec
        )
        output_spec = self.network.output_spec()
        if output_spec.type != 'float':
            raise TensorforceError.type(
                name='ParametrizedStateValue', argument='network output', dtype=output_spec.type
            )

        # State value
        self.value = self.submodule(
            name='value', module='linear', modules=layer_modules, size=0, input_spec=output_spec
        )

    @property
    def internals_spec(self):
        return self.network.internals_spec

    def internals_init(self):
        return self.network.internals_init()

    def max_past_horizon(self, *, on_policy):
        return self.network.max_past_horizon(on_policy=on_policy)

    def get_savedmodel_trackables(self):
        trackables = dict()
        for variable in self.network.variables:
            assert variable.name not in trackables
            trackables[variable.name] = variable
        return trackables

    @tf_function(num_args=0)
    def past_horizon(self, *, on_policy):
        return self.network.past_horizon(on_policy=on_policy)

    @tf_function(num_args=5)
    def next_internals(self, *, states, horizons, internals, actions, deterministic, independent):
        _, internals = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=independent
        )

        return internals

    @tf_function(num_args=4)
    def state_value(self, *, states, horizons, internals, auxiliaries):
        deterministic = tf_util.constant(value=True, dtype='bool')
        embedding, _ = self.network.apply(
            x=states, horizons=horizons, internals=internals, deterministic=deterministic,
            independent=True
        )

        return self.value.apply(x=embedding)
