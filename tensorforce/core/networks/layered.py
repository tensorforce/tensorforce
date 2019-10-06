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

from tensorforce import TensorforceError
from tensorforce.core import Module
from tensorforce.core.layers import layer_modules, StatefulLayer
from tensorforce.core.networks import LayerbasedNetwork


class LayeredNetwork(LayerbasedNetwork):
    """
    Network consisting of Tensorforce layers, which can be specified as either a list of layer
    specifications in the case of a standard sequential layer-stack architecture, or as a list of
    list of layer specifications in the case of a more complex architecture consisting of multiple
    sequential layer-stacks (specification key: `custom` or `layered`).

    Args:
        name (string): Network name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        layers (iter[specification] | iter[iter[specification]]): Layers configuration, see
            [layers](../modules/layers.html)
            (<span style="color:#C00000"><b>required</b></span>).
        inputs_spec (specification): Input tensors specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    # (requires layers as first argument)
    def __init__(
        self, name, layers, inputs_spec, device=None, summary_labels=None, l2_regularization=None
    ):
        super().__init__(
            name=name, inputs_spec=inputs_spec, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.layers_spec = layers

        self.parse_layers_spec(layers_spec=self.layers_spec, layer_counter=Counter())

    def parse_layers_spec(self, layers_spec, layer_counter):
        if isinstance(layers_spec, list):
            for spec in layers_spec:
                self.parse_layers_spec(layers_spec=spec, layer_counter=layer_counter)

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

            # layer_name = self.name + '-' + layer_name
            self.add_module(name=layer_name, module=layers_spec)

    # (requires layers as first argument)
    @classmethod
    def internals_spec(cls, layers=None, network=None, name=None, **kwargs):
        internals_spec = super().internals_spec(network=network)

        if network is None:
            assert layers is not None and name is not None

            for internal_name, spec in cls.internals_from_layers_spec(
                layers_spec=layers, layer_counter=Counter()
            ):
                internal_name = name + '-' + internal_name
                if internal_name in internals_spec:
                    raise TensorforceError.unexpected()
                internals_spec[internal_name] = spec

        else:
            assert layers is None and name is None

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

            layer_cls, first_arg, kwargs = Module.get_module_class_and_kwargs(
                name=layer_name, module=layers_spec, modules=layer_modules
            )
            if issubclass(layer_cls, StatefulLayer):
                if first_arg is None:
                    internals_spec = layer_cls.internals_spec(**kwargs)
                else:
                    internals_spec = layer_cls.internals_spec(first_arg, **kwargs)
                for name, spec in internals_spec.items():
                    name = '{}-{}'.format(layer_name, name)
                    yield name, spec

    def tf_apply(self, x, internals, return_internals=False):
        super().tf_apply(x=x, internals=internals, return_internals=return_internals)

        if isinstance(x, dict):
            x = x[next(iter(x))]

        next_internals = OrderedDict()
        for layer in self.modules.values():
            if isinstance(layer, StatefulLayer):
                layer_internals = {
                    name: internals['{}-{}-{}'.format(self.name, layer.name, name)]
                    for name in layer.__class__.internals_spec(layer=layer)
                }
                assert len(layer_internals) > 0
                x, layer_internals = layer.apply(x=x, initial=layer_internals)
                for name, internal in layer_internals.items():
                    next_internals['{}-{}-{}'.format(self.name, layer.name, name)] = internal

            else:
                x = layer.apply(x=x)

        if return_internals:
            return x, next_internals
        else:
            return x
