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

from tensorforce.core import parameter_modules, tf_function, tf_util
from tensorforce.core.optimizers import UpdateModifier


class ClippingStep(UpdateModifier):
    """
    Clipping-step update modifier, which clips the updates of the given optimizer (specification
    key: `clipping_step`).

    Args:
        optimizer (specification): Optimizer configuration
            (<span style="color:#C00000"><b>required</b></span>).
        threshold (parameter, float >= 0.0): Clipping threshold
            (<span style="color:#C00000"><b>required</b></span>).
        mode ('global_norm' | 'norm' | 'value'): Clipping mode
            (<span style="color:#00C000"><b>default</b></span>: 'global_norm').
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): (<span style="color:#0000C0"><b>internal use</b></span>).
        arguments_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, optimizer, threshold, mode='global_norm', summary_labels=None, name=None,
        arguments_spec=None
    ):
        super().__init__(
            optimizer=optimizer, summary_labels=summary_labels, name=name,
            arguments_spec=arguments_spec
        )

        self.threshold = self.add_module(
            name='threshold', module=threshold, modules=parameter_modules, dtype='float',
            min_value=0.0
        )

        assert mode in ('global_norm', 'norm', 'value')
        self.mode = mode

    @tf_function(num_args=1)
    def step(self, *, arguments, variables, **kwargs):
        deltas = self.optimizer.step(arguments=arguments, variables=variables, **kwargs)

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

            assignments = list()
            for variable, delta, clipped_delta in zip(variables, deltas, clipped_deltas):
                assignments.append(
                    variable.assign_add(delta=(clipped_delta - delta), read_value=False)
                )

        with tf.control_dependencies(control_inputs=assignments):
            return [tf_util.identity(input=delta) for delta in clipped_deltas]
