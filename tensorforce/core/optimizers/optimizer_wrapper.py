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
from tensorforce.core import tf_function
from tensorforce.core.optimizers import UpdateModifier


class OptimizerWrapper(UpdateModifier):
    """
    Optimizer wrapper, which performs additional update modifications, argument order indicates
    modifier nesting from outside to inside
    (specification key: `optimizer_wrapper`).

    Args:
        optimizer (specification): Optimizer
            (<span style="color:#C00000"><b>required</b></span>).
        learning_rate (parameter, float > 0.0): Learning rate
            (<span style="color:#00C000"><b>default</b></span>: 1e-3).
        clipping_threshold (parameter, float > 0.0): Clipping threshold
            (<span style="color:#00C000"><b>default</b></span>: no clipping).
        multi_step (parameter, int >= 1): Number of optimization steps
            (<span style="color:#00C000"><b>default</b></span>: single step).
        subsampling_fraction (parameter, int > 0 | 0.0 < float <= 1.0): Absolute/relative fraction
            of batch timesteps to subsample
            (<span style="color:#00C000"><b>default</b></span>: no subsampling).
        linesearch_iterations (parameter, int >= 0): Maximum number of line search iterations, using
            a backtracking factor of 0.75
            (<span style="color:#00C000"><b>default</b></span>: no line search).
        doublecheck_update (bool): Check whether update has decreased loss and otherwise reverse it
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, optimizer, *, learning_rate=1e-3, clipping_threshold=None, multi_step=1,
        subsampling_fraction=1.0, linesearch_iterations=0, doublecheck_update=False, name=None,
        arguments_spec=None,
        # Deprecated
        optimizing_iterations=None, **kwargs
    ):
        if optimizing_iterations is not None:
            raise TensorforceError.deprecated(
                name='Optimizer', argument='optimizing_iterations',
                replacement='linesearch_iterations'
            )

        if isinstance(optimizer, dict):
            if 'learning_rate' not in optimizer:
                optimizer['learning_rate'] = learning_rate
        else:
            optimizer = dict(type=optimizer)
            optimizer['learning_rate'] = learning_rate

        optimizer.update(kwargs)

        if doublecheck_update:
            optimizer = dict(type='doublecheck_step', optimizer=optimizer)

        if not isinstance(linesearch_iterations, int) or linesearch_iterations > 0:
            optimizer = dict(
                type='linesearch_step', optimizer=optimizer, max_iterations=linesearch_iterations
            )

        if not isinstance(subsampling_fraction, float) or subsampling_fraction != 1.0:
            optimizer = dict(
                type='subsampling_step', optimizer=optimizer, fraction=subsampling_fraction
            )

        if not isinstance(multi_step, int) or multi_step > 1:
            optimizer = dict(type='multi_step', optimizer=optimizer, num_steps=multi_step)

        if clipping_threshold is not None:
            optimizer = dict(
                type='clipping_step', optimizer=optimizer, threshold=clipping_threshold
            )

        super().__init__(optimizer=optimizer, name=name, arguments_spec=arguments_spec)

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        return self.optimizer.step(arguments=arguments, variables=variables, **kwargs)
