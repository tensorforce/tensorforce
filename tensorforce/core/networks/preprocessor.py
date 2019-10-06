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

from collections import Counter, OrderedDict

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core.layers import PreprocessingLayer, StatefulLayer, TemporalLayer
from tensorforce.core.networks import LayerbasedNetwork


class Preprocessor(LayerbasedNetwork):
    """
    Special preprocessor network following a sequential layer-stack architecture, which can be
    specified as either a single or a list of layer specifications.

    Args:
        name (string): Network name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        input_spec (specification): Input tensor specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        layers (iter[specification] | iter[iter[specification]]): Layers configuration, see
            [layers](../modules/layers.html)
            (<span style="color:#C00000"><b>required</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, input_spec, layers, device=None, summary_labels=None, l2_regularization=None
    ):
        super().__init__(
            name=name, inputs_spec=input_spec, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        if isinstance(layers, (dict, str)):
            layers = [layers]

        layer_counter = Counter()
        for layer_spec in layers:
            if 'name' in layer_spec:
                layer_name = layer_spec['name']
            else:
                if isinstance(layer_spec, dict) and isinstance(layer_spec.get('type'), str):
                    layer_type = layer_spec['type']
                else:
                    layer_type = 'layer'
                layer_name = layer_type + str(layer_counter[layer_type])
                layer_counter[layer_type] += 1

            # layer_name = self.name + '-' + layer_name
            self.add_module(name=layer_name, module=layer_spec)

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        raise NotImplementedError

    def internals_init(self):
        raise NotImplementedError

    def tf_dependency_horizon(self, is_optimization=False):
        raise NotImplementedError

    def add_module(self, *args, **kwargs):
        layer = super().add_module(*args, **kwargs)

        if isinstance(layer, (TemporalLayer, StatefulLayer)):
            raise TensorforceError.type(
                name='preprocessor network', argument='sub-module', value=layer
            )

        return layer

    def tf_reset(self):
        operations = list()
        for layer in self.modules.values():
            if isinstance(layer, PreprocessingLayer):
                operations.append(layer.reset())
        return tf.group(*operations)

    def tf_apply(self, x):
        for layer in self.modules.values():
            x = layer.apply(x=x)
        return x

    def create_tf_function(self, name, tf_function):
        if tf_function.__name__ != 'tf_apply':
            return super().create_tf_function(name=name, tf_function=tf_function)

        def validated_tf_function(x):
            if util.is_atomic_values_spec(values_spec=self.inputs_spec):
                if not util.is_consistent_with_value_spec(value_spec=self.inputs_spec, x=x):
                    raise TensorforceError("Invalid input arguments for tf_apply.")
            else:
                if not all(
                    util.is_consistent_with_value_spec(value_spec=spec, x=x[name])
                    for name, spec in self.inputs_spec.items()
                ):
                    raise TensorforceError("Invalid input arguments for tf_apply.")

            x = tf_function(x=x)

            if not util.is_consistent_with_value_spec(value_spec=self.get_output_spec(), x=x):
                raise TensorforceError("Invalid output arguments for tf_apply.")

            return x

        return super().create_tf_function(name=name, tf_function=validated_tf_function)
