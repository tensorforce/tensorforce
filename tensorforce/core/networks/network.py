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

from collections import OrderedDict

from tensorforce import TensorforceError, util
from tensorforce.core import Module
from tensorforce.core.layers import InternalLayer, Layer, layer_modules
from tensorforce.core.parameters import Parameter


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

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        return OrderedDict()

    def internals_init(self):
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
        if tf_function.__name__ != 'tf_apply':
            return super().create_tf_function(name=name, tf_function=tf_function)

        def validated_tf_function(x, internals, return_internals=False):
            if util.is_atomic_values_spec(values_spec=self.inputs_spec):
                if not util.is_consistent_with_value_spec(value_spec=self.inputs_spec, x=x):
                    raise TensorforceError("Invalid input arguments for tf_apply.")
            else:
                if not all(
                    util.is_consistent_with_value_spec(value_spec=spec, x=x[name])
                    for name, spec in self.inputs_spec.items()
                ):
                    raise TensorforceError("Invalid input arguments for tf_apply.")
            if not all(
                util.is_consistent_with_value_spec(value_spec=spec, x=internals[name])
                for name, spec in self.__class__.internals_spec(network=self).items()
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
                for name, spec in self.__class__.internals_spec(network=self).items()
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

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        internals_spec = super().internals_spec()

        if network is not None:
            for layer in network.modules.values():
                if not isinstance(layer, InternalLayer):
                    continue
                for name, spec in layer.__class__.internals_spec(layer=layer).items():
                    name = '{}-{}-{}'.format(network.name, layer.name, name)
                    if name in internals_spec:
                        raise TensorforceError.unexpected()
                    internals_spec[name] = spec

        return internals_spec

    def internals_init(self):
        internals_init = super().internals_init()

        for layer in self.modules.values():
            if not isinstance(layer, InternalLayer):
                continue
            for name, internal_init in layer.internals_init().items():
                internals_init['{}-{}-{}'.format(self.name, layer.name, name)] = internal_init

        return internals_init

    def add_module(self, *args, **kwargs):
        if 'input_spec' in kwargs:
            layer = super().add_module(*args, modules=layer_modules, **kwargs)
            self.output_spec = layer.output_spec

        else:
            if self.output_spec is None:
                if util.is_atomic_values_spec(values_spec=self.inputs_spec):
                    self.output_spec = self.inputs_spec
                elif len(self.inputs_spec) == 1:
                    self.output_spec = next(iter(self.inputs_spec.values()))
                else:
                    self.output_spec = dict(type=None, shape=None)

            layer = super().add_module(
                *args, modules=layer_modules, input_spec=self.output_spec, **kwargs
            )
            self.output_spec = layer.output_spec

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(
                name='layer-based network', argument='sub-module', value=layer
            )

        return layer
