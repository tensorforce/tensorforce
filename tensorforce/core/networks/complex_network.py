# Copyright 2017 reinforce.io. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import json
import os

import tensorflow as tf

from tensorforce import TensorForceError
from tensorforce.core.networks import Layer
from tensorforce.core.networks.network import LayerBasedNetwork


class Input(Layer):
    """
    Input layer. Used for ComplexLayerNetwork's to collect data together
    as a form of output to the next layer.  Allows for multiple inputs
    to merge into a single import for next layer.
    """

    def __init__(
        self,
        inputs,
        axis=1,
        scope='merge_inputs',
        summary_labels=()
    ):
        """
        Input layer.

        Args:
            inputs: A list of strings that name the inputs to merge
            axis: Axis to merge the inputs

        """
        self.inputs = inputs
        self.axis = axis
        super(Input, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        inputs_to_merge = list()
        for name in self.inputs:
            # Previous input, by name or "*", like normal network_spec
            # Not using named_tensors as there could be unintended outcome
            if name == "*" or name == "previous":
                inputs_to_merge.append(x)
            elif name in self.named_tensors:
                inputs_to_merge.append(self.named_tensors[name])
            else:
                # Failed to find key in available inputs, print out help to user, raise error
                keys = list(self.named_tensors)
                raise TensorForceError(
                    'ComplexNetwork input "{}" doesn\'t exist, Available inputs: {}'.format(name, keys)
                )
        # Review data for casting to more precise format so TensorFlow doesn't throw error for mixed data
        # Quick & Dirty cast only promote types: bool=0,int32=10, int64=20, float32=30, double=40

        cast_type_level = 0
        cast_type_dict = {
            'bool': 0,
            'int32': 10,
            'int64': 20,
            'float32': 30,
            'float64': 40
        }
        cast_type_func_dict = {
            0: tf.identity,
            10: tf.to_int32,
            20: tf.to_int64,
            30: tf.to_float,
            40: tf.to_double
        }
        # Scan inputs for max cast_type
        for tensor in inputs_to_merge:
            key = str(tensor.dtype.name)
            if key in cast_type_dict:
                if cast_type_dict[key] > cast_type_level:
                    cast_type_level = cast_type_dict[key]
            else:
                raise TensorForceError('Network spec input does not support dtype {}'.format(key))

        # Add casting if needed
        for index, tensor in enumerate(inputs_to_merge):
            key = str(tensor.dtype.name)

            if cast_type_dict[key] < cast_type_level:
                inputs_to_merge[index] = cast_type_func_dict[cast_type_level](tensor)

        input_tensor = tf.concat(inputs_to_merge, self.axis)

        return input_tensor


class Output(Layer):
    """
    Output layer. Used for ComplexLayerNetwork's to capture the tensor
    under and name for use with Input layers.  Acts as a input to output passthrough.
    """

    def __init__(
        self,
        output,
        scope='output',
        summary_labels=()
    ):
        """
        Output layer.

        Args:
            output: A string that names the tensor, will be added to available inputs

        """
        self.output = output
        super(Output, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        self.named_tensors[self.output] = x
        return x


class ComplexLayeredNetwork(LayerBasedNetwork):
    """
    Complex Network consisting of a sequence of layers, which can be created from a specification dict.
    """

    def __init__(self, complex_layers_spec, scope='layered-network', summary_labels=()):
        """
        Complex Layered network.

        Args:
            complex_layers_spec: List of layer specification dicts
        """
        super(ComplexLayeredNetwork, self).__init__(scope=scope, summary_labels=summary_labels)
        self.complex_layers_spec = complex_layers_spec
        #self.named_tensors = dict()

        layer_counter = Counter()

        for branch_spec in self.complex_layers_spec:
            for layer_spec in branch_spec:
                if isinstance(layer_spec['type'], str):
                    name = layer_spec['type']
                else:
                    name = 'layer'
                scope = name + str(layer_counter[name])
                layer_counter[name] += 1

                layer = Layer.from_spec(
                    spec=layer_spec,
                    kwargs=dict(scope=scope, summary_labels=summary_labels)
                )
                # Link named dictionary reference into Layer.
                layer.tf_tensors(named_tensors=self.named_tensors)
                self.add_layer(layer=layer)

    def tf_apply(self, x, internals, update, return_internals=False):
        if isinstance(x, dict):
            self.named_tensors.update(x)
            if len(x) == 1:
                x = next(iter(x.values()))

        next_internals = dict()
        for layer in self.layers:
            layer_internals = {name: internals['{}_{}'.format(layer.scope, name)] for name in layer.internals_spec()}

            if len(layer_internals) > 0:
                x, layer_internals = layer.apply(x=x, update=update, **layer_internals)
                for name, internal in layer_internals.items():
                    next_internals['{}_{}'.format(layer.scope, name)] = internal

            else:
                x = layer.apply(x=x, update=update)

        if return_internals:
            return x, next_internals
        else:
            return x

    @staticmethod
    def from_json(filename):  # TODO: NOT TESTED
        """
        Creates a complex_layered_network_builder from a JSON.

        Args:
            filename: Path to configuration

        Returns: A ComplexLayeredNetwork class with layers generated from the JSON
        """
        path = os.path.join(os.getcwd(), filename)
        with open(path, 'r') as fp:
            config = json.load(fp=fp)
        return ComplexLayeredNetwork(complex_layers_spec=config)
