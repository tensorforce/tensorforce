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
from tensorforce.core.optimizers import MetaOptimizer


class ClippingStep(MetaOptimizer):
    """
    Clipping-step meta optimizer, which clips the updates of the given optimizer (specification
    key: `clipping_step`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        threshold (parameter, float > 0.0): Clipping threshold
            (<span style="color:#C00000"><b>required</b></span>).
        mode ('global_norm' | 'norm' | 'value'): Clipping mode
            (<span style="color:#00C000"><b>default</b></span>: 'global_norm').
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, optimizer, threshold, mode='global_norm', summary_labels=None):
        super().__init__(name=name, optimizer=optimizer, summary_labels=summary_labels)

        self.threshold = self.add_module(
            name='threshold', module=threshold, modules=parameter_modules, dtype='float'
        )

        assert mode in ('global_norm', 'norm', 'value')
        self.mode = mode

    def tf_step(self, variables, **kwargs):
        deltas = self.optimizer.step(variables=variables, **kwargs)

        with tf.control_dependencies(control_inputs=deltas):
            threshold = self.threshold.value()
            if self.mode == 'global_norm':
                clipped_deltas, update_norm = tf.clip_by_global_norm(
                    t_list=deltas, clip_norm=threshold
                )
            else:
                update_norm = tf.linalg.global_norm(t_list=deltas)
                clipped_deltas = list()
                for delta in deltas:
                    if self.mode == 'norm':
                        clipped_delta = tf.clip_by_norm(t=delta, clip_norm=threshold)
                    elif self.mode == 'value':
                        clipped_delta = tf.clip_by_value(
                            t=delta, clip_value_min=-threshold, clip_value_max=threshold
                        )
                    clipped_deltas.append(clipped_delta)

            clipped_deltas = self.add_summary(
                label='update-norm', name='update-norm-unclipped', tensor=update_norm,
                pass_tensors=clipped_deltas
            )

            exceeding_deltas = list()
            for delta, clipped_delta in zip(deltas, clipped_deltas):
                exceeding_deltas.append(clipped_delta - delta)

        applied = self.apply_step(variables=variables, deltas=exceeding_deltas)

        with tf.control_dependencies(control_inputs=(applied,)):
            return util.fmap(function=util.identity_operation, xs=clipped_deltas)
