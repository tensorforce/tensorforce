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

import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import Module
from tensorforce.core.layers import Layer, layer_modules, StatefulLayer, TemporalLayer
from tensorforce.core.parameters import Parameter


class Network(Module):
    """
    Base class for neural networks.

    Args:
        name (string): Network name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        inputs_spec (specification): Input tensors specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, inputs_spec, device=None, summary_labels=None, l2_regularization=None
    ):
        super().__init__(
            name=name, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.inputs_spec = inputs_spec

    def get_output_spec(self):
        raise NotImplementedError

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        return OrderedDict()

    def internals_init(self):
        return OrderedDict()

    def tf_dependency_horizon(self, is_optimization=False):
        return tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))

    def tf_apply(self, x, internals, return_internals=False):
        Module.update_tensors(**x)

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
    Base class for networks using Tensorforce layers.
    """

    def __init__(
        self, name, inputs_spec, device=None, summary_labels=None, l2_regularization=None
    ):
        """
        Layer-based network constructor.
        """
        super().__init__(
            name=name, inputs_spec=inputs_spec, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        if len(inputs_spec) == 1:
            self.output_spec = next(iter(inputs_spec.values()))
        else:
            self.output_spec = None

    def get_output_spec(self):
        return self.output_spec

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        internals_spec = OrderedDict()

        if network is not None:
            for layer in network.modules.values():
                if not isinstance(layer, StatefulLayer):
                    continue
                for name, spec in layer.__class__.internals_spec(layer=layer).items():
                    name = '{}-{}-{}'.format(network.name, layer.name, name)
                    if name in internals_spec:
                        raise TensorforceError.unexpected()
                    internals_spec[name] = spec

        return internals_spec

    def internals_init(self):
        internals_init = OrderedDict()

        for layer in self.modules.values():
            if not isinstance(layer, StatefulLayer):
                continue
            for name, internal_init in layer.internals_init().items():
                internals_init['{}-{}-{}'.format(self.name, layer.name, name)] = internal_init

        return internals_init

    def add_module(self, *args, **kwargs):
        # Default modules set: layer_modules
        if len(args) < 3 and 'modules' not in kwargs:
            assert 'is_subscope' not in kwargs
            kwargs['modules'] = layer_modules
            kwargs['is_subscope'] = True

        if 'input_spec' in kwargs:
            layer = super().add_module(*args, **kwargs)
            self.output_spec = layer.output_spec

        else:
            if self.output_spec is None:
                if util.is_atomic_values_spec(values_spec=self.inputs_spec):
                    self.output_spec = self.inputs_spec
                elif len(self.inputs_spec) == 1:
                    self.output_spec = next(iter(self.inputs_spec.values()))
                else:
                    self.output_spec = None

            if self.output_spec is not None:
                if 'input_spec' in kwargs:
                    kwargs['input_spec'] = util.unify_value_specs(
                        value_spec1=kwargs['input_spec'], value_spec2=self.output_spec
                    )
                else:
                    kwargs['input_spec'] = self.output_spec

            layer = super().add_module(*args, **kwargs)

            self.output_spec = layer.output_spec

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(
                name='layer-based network', argument='sub-module', value=layer
            )

        return layer

    def tf_dependency_horizon(self, is_optimization=False):
        dependencies = [super().tf_dependency_horizon()]
        for layer in self.modules.values():
            if isinstance(layer, TemporalLayer):
                if not isinstance(layer, StatefulLayer) or is_optimization:
                    dependencies.append(layer.dependency_horizon.value())

        return tf.math.reduce_max(input_tensor=tf.stack(values=dependencies, axis=0), axis=0)
