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

from tensorforce import TensorforceError
from tensorforce.core import tf_util
from tensorforce.core.parameters import Decaying


class Exponential(Decaying):
    """
    Exponentially decaying hyperparameter (specification key: `exponential`).

    Args:
        unit ("timesteps" | "episodes" | "updates"): Unit of decay schedule
            (<span style="color:#C00000"><b>required</b></span>).
        num_steps (int): Number of decay steps
            (<span style="color:#C00000"><b>required</b></span>).
        initial_value (float): Initial value
            (<span style="color:#C00000"><b>required</b></span>).
        decay_rate (float): Decay rate
            (<span style="color:#C00000"><b>required</b></span>).
        staircase (bool): Whether to apply decay in a discrete staircase, as opposed to continuous,
            fashion (<span style="color:#00C000"><b>default</b></span>: false).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        dtype (type): <span style="color:#0000C0"><b>internal use</b></span>.
        min_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
        max_value (dtype-compatible value): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, unit, num_steps, initial_value, decay_rate, staircase=False, name=None, dtype=None,
        min_value=None, max_value=None, **kwargs
    ):
        super().__init__(
            decay='exponential', unit=unit, num_steps=num_steps, initial_value=initial_value,
            name=name, dtype=dtype, min_value=min_value, max_value=max_value, decay_rate=decay_rate,
            staircase=staircase, **kwargs
        )
