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

from collections import Counter

import tensorflow as tf

from tensorforce import TensorforceError
from tensorforce.core import ArrayDict, Module, SignatureDict, TensorDict, TensorSpec, \
    TensorsSpec, tf_function, tf_util
from tensorforce.core.layers import Layer, layer_modules, MultiInputLayer, NondeterministicLayer, \
    PreprocessingLayer, Register, StatefulLayer, TemporalLayer
from tensorforce.core.parameters import Parameter


class Network(Module):
    """
    Base class for neural networks.

    Args:
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        inputs_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(self, *, device=None, l2_regularization=None, name=None, inputs_spec=None):
        super().__init__(name=name, device=device,  l2_regularization=l2_regularization)

        self.inputs_spec = inputs_spec

    def output_spec(self):
        raise NotImplementedError

    @property
    def internals_spec(self):
        return TensorsSpec()

    def internals_init(self):
        return ArrayDict()

    def max_past_horizon(self, *, on_policy):
        return 0

    def input_signature(self, *, function):
        if function == 'apply':
            return SignatureDict(
                x=self.inputs_spec.signature(batched=True),
                horizons=TensorSpec(type='int', shape=(2,)).signature(batched=True),
                internals=self.internals_spec.signature(batched=True),
                deterministic=TensorSpec(type='bool', shape=()).signature(batched=False)
            )

        elif function == 'past_horizon':
            return SignatureDict()

        else:
            return super().input_signature(function=function)

    def output_signature(self, *, function):
        if function == 'apply':
            return SignatureDict(
                x=self.output_spec().signature(batched=True),
                internals=self.internals_spec.signature(batched=True)
            )

        elif function == 'past_horizon':
            return SignatureDict(
                singleton=TensorSpec(type='int', shape=()).signature(batched=False)
            )

        else:
            return super().output_signature(function=function)

    @tf_function(num_args=0)
    def past_horizon(self, *, on_policy):
        return tf_util.constant(value=0, dtype='int')

    @tf_function(num_args=3)
    def apply(self, *, x, horizons, internals, independent):
        raise NotImplementedError


class LayerbasedNetwork(Network):
    """
    Base class for networks using Tensorforce layers.
    """

    def __init__(self, *, name, inputs_spec, device=None, l2_regularization=None):
        super().__init__(
            name=name, inputs_spec=inputs_spec, device=device, l2_regularization=l2_regularization
        )

        if self.inputs_spec.is_singleton():
            self.registered_tensors_spec = TensorsSpec(state=self.inputs_spec.singleton())
        else:
            self.registered_tensors_spec = self.inputs_spec.copy()

        self._output_spec = self.inputs_spec.value()

    def invalid_layer_types(self):
        return (PreprocessingLayer,)

    def output_spec(self):
        return self._output_spec

    @property
    def internals_spec(self):
        internals_spec = super().internals_spec

        for layer in self.this_submodules:
            if isinstance(layer, TemporalLayer):
                internals_spec[layer.name] = layer.internals_spec

        return internals_spec

    def internals_init(self):
        internals_init = super().internals_init()

        for layer in self.this_submodules:
            if isinstance(layer, TemporalLayer):
                internals_init[layer.name] = layer.internals_init()

        return internals_init

    def max_past_horizon(self, *, on_policy):
        past_horizons = [super().max_past_horizon(on_policy=on_policy)]

        for layer in self.this_submodules:
            if isinstance(layer, TemporalLayer):
                past_horizons.append(layer.max_past_horizon(on_policy=on_policy))

        return max(past_horizons)

    @tf_function(num_args=0)
    def past_horizon(self, *, on_policy):
        past_horizons = [super().past_horizon(on_policy=on_policy)]

        for layer in self.this_submodules:
            if isinstance(layer, TemporalLayer):
                past_horizons.append(layer.past_horizon(on_policy=on_policy))

        return tf.math.reduce_max(input_tensor=tf.stack(values=past_horizons, axis=0), axis=0)

    def submodule(
        self, *, name, module=None, modules=None, default_module=None, is_trainable=True,
        is_saved=True, **kwargs
    ):
        # Module class and args
        if modules is None:
            modules = layer_modules
        module_cls, args, kwargs = Module.get_module_class_and_args(
            name=name, module=module, modules=modules, default_module=default_module, **kwargs
        )
        if len(args) > 0:
            assert len(kwargs) == 0
            module_cls = args[0]

        # Default input_spec
        if not issubclass(module_cls, Layer):
            pass

        elif kwargs.get('input_spec') is None:
            if issubclass(module_cls, MultiInputLayer):
                if 'tensors' not in kwargs:
                    raise TensorforceError.required(name='MultiInputLayer', argument='tensors')
                if tuple(kwargs['tensors']) not in self.registered_tensors_spec:
                    raise TensorforceError.exists_not(
                        name='registered tensor', value=kwargs['tensors']
                    )
                kwargs['input_spec'] = self.registered_tensors_spec[tuple(kwargs['tensors'])]

            elif self._output_spec is None:
                raise TensorforceError.required(
                    name='layer-based network', argument='first layer', expected='retrieve',
                    condition='multiple state/input components'
                )

            else:
                kwargs['input_spec'] = self._output_spec

        elif issubclass(module_cls, MultiInputLayer):
            raise TensorforceError.invalid(name='MultiInputLayer', argument='input_spec')

        layer = super().submodule(
            module=module_cls, modules=modules, default_module=default_module,
            is_trainable=is_trainable, is_saved=is_saved, **kwargs
        )

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(
                name='layer-based network', argument='sub-module', value=layer
            )

        elif isinstance(layer, self.invalid_layer_types()):
            raise TensorforceError.type(
                name='network', argument='layer', value=layer, hint='invalid layer type'
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
    Network consisting of Tensorforce layers (specification key: `custom` or `layered`), which can
    be specified as either a list of layer specifications in the case of a standard sequential
    layer-stack architecture, or as a list of list of layer specifications in the case of a more
    complex architecture consisting of multiple sequential layer-stacks. Note that the final
    action/value layer of the policy/baseline network is implicitly added, so the network output can
    be of arbitrary size and use any activation function, and is only required to be a rank-one
    embedding vector, or optionally have the same shape as the action in the case of a higher-rank
    action shape.

    Args:
        layers (iter[specification] | iter[iter[specification]]): Layers configuration, see the
            [layers documentation](../modules/layers.html)
            (<span style="color:#C00000"><b>required</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        inputs_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    # (requires layers as first argument)
    def __init__(self, layers, *, device=None, l2_regularization=None, name=None, inputs_spec=None):
        super().__init__(
            device=device, l2_regularization=l2_regularization, name=name, inputs_spec=inputs_spec
        )

        self.layers = list(self._parse_layers_spec(spec=layers, counter=Counter()))

    def _parse_layers_spec(self, *, spec, counter):
        if isinstance(spec, list):
            for s in spec:
                yield from self._parse_layers_spec(spec=s, counter=counter)

        else:
            if callable(spec):
                spec = dict(type='function', function=spec)
            elif isinstance(spec, str):
                spec = dict(type=spec)

            # Deprecated
            if spec.get('type') in ('internal_rnn', 'internal_lstm', 'internal_gru'):
                raise TensorforceError.deprecated(
                    name='Network layers', argument=spec['type'], replacement=spec['type'][9:]
                )

            if 'name' in spec:
                spec = dict(spec)
                name = spec.pop('name')

            else:
                layer_type = spec.get('type')
                if not isinstance(layer_type, str):
                    layer_type = 'layer'
                name = layer_type + str(counter[layer_type])
                counter[layer_type] += 1

            yield self.submodule(name=name, module=spec)

    @tf_function(num_args=4)
    def apply(self, *, x, horizons, internals, deterministic, independent):
        if x.is_singleton():
            registered_tensors = TensorDict(state=x.singleton())
        else:
            registered_tensors = x.copy()
        x = x.value()

        for layer in self.layers:
            if isinstance(layer, Register):
                if layer.tensor in registered_tensors:
                    raise TensorforceError.exists(name='registered tensor', value=layer.tensor)
                x = layer.apply(x=x)
                registered_tensors[layer.tensor] = x

            elif isinstance(layer, MultiInputLayer):
                if layer.tensors not in registered_tensors:
                    raise TensorforceError.exists_not(name='registered tensor', value=layer.tensors)
                x = layer.apply(x=registered_tensors[layer.tensors])

            elif isinstance(layer, NondeterministicLayer):
                x = layer.apply(x=x, deterministic=deterministic)

            elif isinstance(layer, StatefulLayer):
                x = layer.apply(x=x, independent=independent)

            elif isinstance(layer, TemporalLayer):
                x, internals[layer.name] = layer.apply(
                    x=x, horizons=horizons, internals=internals[layer.name]
                )

            else:
                x = layer.apply(x=x)

        return x, internals
