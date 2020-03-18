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
from tensorforce.core import tf_function
from tensorforce.core.layers import PreprocessingLayer, TemporalLayer
from tensorforce.core.networks import LayerbasedNetwork


class Preprocessor(LayerbasedNetwork):
    """
    Special preprocessor network following a sequential layer-stack architecture, which can be
    specified as either a single or a list of layer specifications.

    Args:
        layers (iter[specification] | iter[iter[specification]]): Layers configuration, see
            [layers](../modules/layers.html)
            (<span style="color:#C00000"><b>required</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        inputs_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, layers, device=None, summary_labels=None, l2_regularization=None, name=None,
        inputs_spec=None
    ):
        super().__init__(
            device=device, summary_labels=summary_labels, l2_regularization=l2_regularization,
            name=name, inputs_spec=inputs_spec
        )

        if isinstance(layers, (dict, str)):
            layers = [layers]

        self.layers = list()
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
            self.layers.append(self.add_module(name=layer_name, module=layer_spec))

    def input_signature(self, function):
        if function == 'apply':
            return [util.to_tensor_spec(value_spec=self.inputs_spec, batched=True)]

        else:
            return super().input_signature(function=function)

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        return None

    def internals_init(self):
        raise NotImplementedError

    def add_module(self, *args, **kwargs):
        layer = super().add_module(*args, **kwargs)

        if isinstance(layer, TemporalLayer):
            raise TensorforceError.type(
                name='preprocessor network', argument='sub-module', value=layer
            )

        return layer

    @tf_function(num_args=0)
    def reset(self):
        operations = list()
        for layer in self.layers:
            if isinstance(layer, PreprocessingLayer):
                operations.append(layer.reset())
        return tf.group(*operations)

    @tf_function(num_args=1)
    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x=x)
        return x
