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

from tensorforce.core import tf_function
from tensorforce.core.optimizers import UpdateModifier


class UpdateModifierWrapper(UpdateModifier):
    """
    Update modifier wrapper (specification key: `update_modifier_wrapper`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        multi_step (parameter, int > 0): Number of optimization steps
            (<span style="color:#00C000"><b>default</b></span>: single step).
        subsampling_fraction (parameter, 0.0 < float <= 1.0): Fraction of batch timesteps to
            subsample (<span style="color:#00C000"><b>default</b></span>: no subsampling).
        clipping_threshold (parameter, float > 0.0): Clipping threshold
            (<span style="color:#00C000"><b>default</b></span>: no clipping).
        optimizing_iterations (parameter, int >= 0):  Maximum number of line search iterations
            (<span style="color:#00C000"><b>default</b></span>: no optimizing).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, optimizer, *, multi_step=1, subsampling_fraction=1.0, clipping_threshold=None,
        optimizing_iterations=0, name=None, arguments_spec=None, **kwargs
    ):
        optimizer = dict(type=optimizer)
        optimizer.update(kwargs)

        if optimizing_iterations > 0:
            optimizer = dict(
                type='optimizing_step', optimizer=optimizer, ls_max_iterations=optimizing_iterations
            )

        if clipping_threshold is not None:
            optimizer = dict(
                type='clipping_step', optimizer=optimizer, threshold=clipping_threshold
            )

        if subsampling_fraction != 1.0:
            optimizer = dict(
                type='subsampling_step', optimizer=optimizer, fraction=subsampling_fraction
            )

        if multi_step > 1:
            optimizer = dict(type='multi_step', optimizer=optimizer, num_steps=multi_step)

        super().__init__(optimizer=optimizer, name=name, arguments_spec=arguments_spec)

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        return self.optimizer.step(arguments=arguments, variables=variables, **kwargs)
