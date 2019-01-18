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

from tensorforce import TensorforceError, util
from tensorforce.core import Module
from tensorforce.core.layers import Layer, layer_modules


class Network(Module):
    """
    Base class for neural networks.
    """

    def __init__(self, name, inputs_spec, l2_regularization=None, summary_labels=None):
        """
        Layer-based network.
        """
        super().__init__(
            name=name, l2_regularization=l2_regularization, summary_labels=summary_labels
        )

        self.inputs_spec = inputs_spec

    def get_output_spec(self):
        raise NotImplementedError

    def internals_spec(self):
        """
        Returns the internal states specification.

        Returns:
            Internal states specification
        """
        return OrderedDict()

    def internals_init(self):
        """
        Returns the internal states initialization tensor.

        Returns:
            Internal states initialization tensor
        """
        return OrderedDict()

    def tf_apply(self, x, internals, return_internals=False):
        """
        Creates the TensorFlow operations for applying the network to the given input.

        Args:
            x: Network input tensor or dict of input tensors.
            internals: List of prior internal state tensors
            return_internals: If true, also returns posterior internal state tensors

        Returns:
            Network output tensor, plus optionally list of posterior internal state tensors
        """
        raise NotImplementedError

    def create_tf_function(self, name, tf_function):
        if name[-6:] != '.apply':
            return super().create_tf_function(name=name, tf_function=tf_function)

        def validated_tf_function(x, internals, return_internals=False):
            if util.is_atomic_values_spec(values_spec=self.inputs_spec):
                if not util.is_consistent_with_value_spec(
                    value_spec=self.inputs_spec, x=x
                ):
                    raise TensorforceError("Invalid input arguments for tf_apply.")
            else:
                if not all(
                    util.is_consistent_with_value_spec(value_spec=spec, x=x[name])
                    for name, spec in self.inputs_spec.items()
                ):
                    raise TensorforceError("Invalid input arguments for tf_apply.")
            if not all(
                util.is_consistent_with_value_spec(value_spec=spec, x=internals[name])
                for name, spec in self.internals_spec().items()
            ):
                raise TensorforceError("Invalid input arguments for tf_apply.")

            if isinstance(x, dict):
                Module.update_tensors(**x)
            Module.update_tensors(**internals)

            if return_internals:
                x, internals = tf_function(x=x, internals=internals, return_internals=True)
            else:
                x = tf_function(x=x, internals=internals, return_internals=False)

            if not util.is_consistent_with_value_spec(value_spec=self.get_output_spec(), x=x):
                raise TensorforceError("Invalid output arguments for tf_apply.")
            if return_internals and not all(
                util.is_consistent_with_value_spec(value_spec=spec, x=internals[name])
                for name, spec in self.internals_spec().items()
            ):
                raise TensorforceError("Invalid output arguments for tf_apply.")

            if return_internals:
                return x, internals
            else:
                return x

        return super().create_tf_function(name=name, tf_function=validated_tf_function)


class LayerbasedNetwork(Network):
    """
    Base class for networks using TensorForce layers.
    """

    def __init__(self, name, inputs_spec, l2_regularization=None, summary_labels=None):
        super().__init__(
            name=name, inputs_spec=inputs_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

        self.output_spec = None

    def get_output_spec(self):
        return self.output_spec

    def internals_spec(self):
        specification = super().internals_spec()

        for layer in self.modules.values():
            for name, spec in layer.internals_spec().items():
                name = layer.name + '-' + name
                if name in specification:
                    raise TensorforceError.unexpected()
                specification[name] = spec

        return specification

    def internals_init(self):
        initialization = super().internals_init()

        for layer in self.modules.values():
            for name, init in layer.internals_init().items():
                initialization[layer.name + '-' + name] = init

        return initialization

    def add_module(self, *args, **kwargs):
        if 'input_spec' in kwargs:
            if kwargs['input_spec'] != self.output_spec:
                raise TensorforceError(message="Unexpected specification mismatch.")

            layer = super().add_module(*args, **kwargs)

        else:
            if self.output_spec is None:
                if util.is_atomic_values_spec(values_spec=self.inputs_spec):
                    self.output_spec = self.inputs_spec
                elif len(self.inputs_spec) == 1:
                    self.output_spec = next(iter(self.inputs_spec.values()))
                else:
                    self.output_spec = dict(type=None, shape=None)

            layer = super().add_module(*args, input_spec=self.output_spec, **kwargs)
            self.output_spec = layer.output_spec

        if not isinstance(layer, Layer):
            raise TensorforceError.type(
                name='layer-based network', argument='sub-module', value=layer
            )

        return layer


class LayeredNetwork(LayerbasedNetwork):
    """
    Network consisting of a sequence of layers, which can be created from a specification dict.
    """

    def __init__(self, name, layers, inputs_spec, l2_regularization=None, summary_labels=None):
        """
        Single-stack layered network.

        Args:
            layers: List of layer specification dicts.
        """
        super().__init__(
            name=name, inputs_spec=inputs_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

        self.layers_spec = layers

        self.parse_layers_spec(layers_spec=self.layers_spec, layer_counter=Counter())

    def parse_layers_spec(self, layers_spec, layer_counter):
        if isinstance(layers_spec, list):
            for spec in layers_spec:
                self.parse_layers_spec(layers_spec=spec, layer_counter=layer_counter)

        else:
            if isinstance(layers_spec['type'], str):
                layer_type = layers_spec['type']
            else:
                layer_type = 'layer'
            name = layer_type + str(layer_counter[layer_type])
            layer_counter[layer_type] += 1

            self.add_module(name=name, module=layers_spec, modules=layer_modules)

    def tf_apply(self, x, internals, return_internals=False):
        if isinstance(x, dict):
            x = x[next(iter(x))]

        next_internals = OrderedDict()
        for layer in self.modules.values():
            layer_internals = {
                name: internals[layer.name + '-' + name] for name in layer.internals_spec()
            }

            if len(layer_internals) > 0:
                x, layer_internals = layer.apply(x=x, **layer_internals)
                for name, internal in layer_internals.items():
                    next_internals[layer.name + '-' + name] = internal

            else:
                x = layer.apply(x=x)

        if return_internals:
            return x, next_internals
        else:
            return x
