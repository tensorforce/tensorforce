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
from tensorforce.core import Module, tf_function
from tensorforce.core.layers import Layer, layer_modules, Register, Retrieve, TemporalLayer
from tensorforce.core.parameters import Parameter


class Network(Module):
    """
    Base class for neural networks.

    Args:
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
        self, device=None, summary_labels=None, l2_regularization=None, name=None, inputs_spec=None
    ):
        super().__init__(
            name=name, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        assert all(
            util.is_atomic_values_spec(values_spec=input_spec)
            for input_spec in inputs_spec.values()
        )
        self.inputs_spec = inputs_spec

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        return OrderedDict()

    def internals_init(self):
        return OrderedDict()

    def output_spec(self):
        raise NotImplementedError

    def max_past_horizon(self, on_policy):
        raise NotImplementedError

    def input_signature(self, function):
        if function == 'apply':
            return [
                util.to_tensor_spec(value_spec=self.inputs_spec, batched=True),
                util.to_tensor_spec(value_spec=dict(type='long', shape=(2,)), batched=True),
                util.to_tensor_spec(
                    value_spec=self.__class__.internals_spec(network=self), batched=True
                )
            ]

        elif function == 'past_horizon':
            return ()

        else:
            return super().input_signature(function=function)

    @tf_function(num_args=0)
    def past_horizon(self, on_policy):
        raise NotImplementedError

    @tf_function(num_args=3)
    def apply(self, x, horizons, internals, return_internals):
        raise NotImplementedError


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

        self.registered_tensors_spec = dict(self.inputs_spec)

        if len(self.inputs_spec) == 1:
            self._output_spec = next(iter(self.inputs_spec.values()))
        else:
            self._output_spec = None

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        internals_spec = super().internals_spec(network=network, **kwargs)

        if network is not None:
            assert len(kwargs) == 0
            for layer in network.this_submodules:
                if isinstance(layer, TemporalLayer):
                    for name, spec in layer.__class__.internals_spec(layer=layer).items():
                        internals_spec['{}-{}-{}'.format(network.name, layer.name, name)] = spec

        return internals_spec

    def internals_init(self):
        internals_init = super().internals_init()

        for layer in self.this_submodules:
            if isinstance(layer, TemporalLayer):
                for name, init in layer.internals_init().items():
                    internals_init['{}-{}-{}'.format(self.name, layer.name, name)] = init

        return internals_init

    def output_spec(self):
        return self._output_spec

    def max_past_horizon(self, on_policy):
        past_horizons = [0]

        for layer in self.this_submodules:
            if isinstance(layer, TemporalLayer):
                past_horizons.append(layer.max_past_horizon(on_policy=on_policy))

        return max(past_horizons)

    @tf_function(num_args=0)
    def past_horizon(self, on_policy):
        past_horizons = [tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))]

        for layer in self.this_submodules:
            if isinstance(layer, TemporalLayer):
                past_horizons.append(layer.past_horizon(on_policy=on_policy))

        return tf.math.reduce_max(input_tensor=tf.stack(values=past_horizons, axis=0), axis=0)

    def add_module(
        self, name, module=None, modules=None, default_module=None, is_subscope=False,
        is_trainable=True, is_saved=True, **kwargs
    ):
        # Default modules
        if modules is None:
            modules = layer_modules

        module, first_arg, kwargs = Module.get_module_class_and_kwargs(
            name=name, module=module, modules=modules, default_module=default_module, **kwargs
        )
        if first_arg is not None:
            assert len(kwargs) == 0
            module = first_arg

        # Default input_spec
        if not issubclass(module, Layer):
            pass

        elif kwargs.get('input_spec') is None:
            if module is Retrieve:
                if 'tensors' not in kwargs:
                    raise TensorforceError.required(name='retrieve layer', argument='tensors')
                kwargs['input_spec'] = OrderedDict()
                for tensor in kwargs['tensors']:
                    if tensor not in self.registered_tensors_spec:
                        raise TensorforceError.exists_not(name='registered tensor', value=tensor)
                    kwargs['input_spec'][tensor] = self.registered_tensors_spec[tensor]
            elif self._output_spec is None:
                raise TensorforceError.required(
                    name='layer-based network', argument='first layer', expected='retrieve',
                    condition='multiple state/input components'
                )
            else:
                kwargs['input_spec'] = self._output_spec

        elif module is Retrieve:
            raise TensorforceError.invalid(name='retrieve layer', argument='input_spec')

        layer = super().add_module(
            module=module, modules=modules, default_module=default_module, is_subscope=is_subscope,
            is_trainable=is_trainable, is_saved=is_saved, **kwargs
        )

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(
                name='layer-based network', argument='sub-module', value=layer
            )

        if isinstance(layer, Layer):
            self._output_spec = layer.output_spec()

            if isinstance(layer, Register):
                if layer.tensor in self.registered_tensors_spec:
                    raise TensorforceError.exists(name='registered tensor', value=layer.tensor)
                self.registered_tensors_spec[layer.tensor] = layer.output_spec()

        return layer


class LayeredNetwork(LayerbasedNetwork):
    """
    Network consisting of Tensorforce layers, which can be specified as either a list of layer
    specifications in the case of a standard sequential layer-stack architecture, or as a list of
    list of layer specifications in the case of a more complex architecture consisting of multiple
    sequential layer-stacks (specification key: `custom` or `layered`).

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

    # (requires layers as first argument)
    def __init__(
        self, layers, device=None, summary_labels=None, l2_regularization=None, name=None,
        inputs_spec=None
    ):
        super().__init__(
            device=device, summary_labels=summary_labels, l2_regularization=l2_regularization,
            name=name, inputs_spec=inputs_spec
        )

        self.layers_spec = layers
        self.layers = list()

        self.parse_layers_spec(
            layers=self.layers, layers_spec=self.layers_spec, layer_counter=Counter()
        )

    def parse_layers_spec(self, layers, layers_spec, layer_counter):
        if isinstance(layers_spec, list):
            for spec in layers_spec:
                self.parse_layers_spec(layers=layers, layers_spec=spec, layer_counter=layer_counter)

        else:
            if 'name' in layers_spec:
                layers_spec = dict(layers_spec)
                layer_name = layers_spec.pop('name')
            else:
                if isinstance(layers_spec.get('type'), str):
                    layer_type = layers_spec['type']
                else:
                    layer_type = 'layer'
                layer_name = layer_type + str(layer_counter[layer_type])
                layer_counter[layer_type] += 1

            layers.append(self.add_module(name=layer_name, module=layers_spec))

    # (requires layers as first argument)
    @classmethod
    def internals_spec(cls, layers=None, network=None, **kwargs):
        assert network is None or layers is None
        internals_spec = super().internals_spec(network=network)

        if network is None:
            assert layers is not None and 'name' in kwargs
            for name, spec in cls.internals_from_layers_spec(
                layers_spec=layers, layer_counter=Counter()
            ).items():
                internals_spec['{}-{}'.format(kwargs['name'], name)] = spec

        return internals_spec

    @classmethod
    def internals_from_layers_spec(cls, layers_spec, layer_counter):
        if isinstance(layers_spec, list):
            for spec in layers_spec:
                yield from cls.internals_from_layers_spec(
                    layers_spec=spec, layer_counter=layer_counter
                )

        else:
            if 'name' in layers_spec:
                layers_spec = dict(layers_spec)
                layer_name = layers_spec.pop('name')
            else:
                if isinstance(layers_spec.get('type'), str):
                    layer_type = layers_spec['type']
                else:
                    layer_type = 'layer'
                layer_name = layer_type + str(layer_counter[layer_type])
                layer_counter[layer_type] += 1

            layer_cls, first_arg, layer_kwargs = Module.get_module_class_and_kwargs(
                name=layer_name, module=layers_spec, modules=layer_modules
            )

            if issubclass(layer_cls, TemporalLayer):
                if first_arg is None:
                    for name, spec in layer_cls.internals_spec(**layer_kwargs).items():
                        yield '{}-{}'.format(layer_name, name), spec
                else:
                    for name, spec in layer_cls.internals_spec(first_arg, **layer_kwargs).items():
                        yield '{}-{}'.format(layer_name, name), spec

    @tf_function(num_args=3)
    def apply(self, x, horizons, internals, return_internals):
        if isinstance(x, dict) and len(x) == 1:
            x = next(iter(x.values()))

        registered_tensors = dict(x)
        next_internals = OrderedDict()
        for layer in self.layers:
            if isinstance(layer, Register):
                if layer.tensor in registered_tensors:
                    raise TensorforceError.exists(name='registered tensor', value=layer.tensor)
                x = layer.apply(x=x)
                registered_tensors[layer.tensor] = x

            elif isinstance(layer, Retrieve):
                x = OrderedDict()
                for tensor in layer.tensors:
                    if tensor not in registered_tensors:
                        raise TensorforceError.exists_not(name='registered tensor', value=tensor)
                    x[tensor] = registered_tensors[tensor]
                x = layer.apply(x=x)

            elif isinstance(layer, TemporalLayer):
                layer_internals = OrderedDict(
                    (name, internals['{}-{}-{}'.format(self.name, layer.name, name)])
                    for name in layer.__class__.internals_spec(layer=layer)
                )
                x, layer_internals = layer.apply(
                    x=x, horizons=horizons, internals=layer_internals
                )
                for name, internal in layer_internals.items():
                    next_internals['{}-{}-{}'.format(self.name, layer.name, name)] = internal

            else:
                x = layer.apply(x=x)

        # starts = tf.range(start=batch_size, dtype=util.tf_dtype(dtype='long'))
        # lengths = tf.ones(shape=(batch_size,), dtype=util.tf_dtype(dtype='long'))
        # empty_horizons = tf.concat(values=(starts, lengths))
        # assertion = tf.debugging.assert_equal(x=horizons, y=empty_horizons, axis=1)
        # with control_dependencies(control_inputs=(assertion,)):
        #     x = util.identity_operation(x=x)

        if return_internals:
            return x, next_internals
        else:
            return x
