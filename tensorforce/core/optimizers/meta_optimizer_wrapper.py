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

from tensorforce.core.optimizers import MetaOptimizer


class MetaOptimizerWrapper(MetaOptimizer):
    """
    Meta optimizer wrapper
    """

    def __init__(
        self, name, optimizer, multi_step=1, subsampling_fraction=1.0, clipping_value=None,
        optimized_iterations=0, summary_labels=None, **kwargs
    ):
        optimizer = dict(type=optimizer)
        optimizer.update(kwargs)
        if optimized_iterations > 0:
            optimizer = dict(
                type='optimized_step', optimizer=optimizer, ls_max_iterations=optimized_iterations
            )
        if clipping_value is not None:
            optimizer = dict(
                type='clipped_step', optimizer=optimizer, clipping_value=clipping_value
            )
        if subsampling_fraction != 1.0:
            optimizer = dict(
                type='subsampling_step', optimizer=optimizer, fraction=subsampling_fraction
            )
        if multi_step > 1:
            optimizer = dict(type='multi_step', optimizer=optimizer, num_steps=multi_step)

        super().__init__(name=name, optimizer=optimizer, summary_labels=summary_labels)

    def tf_step(self, variables, **kwargs):
        return self.optimizer.step(variables=variables, **kwargs)
