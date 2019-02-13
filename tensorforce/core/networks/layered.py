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
from tensorforce.core.layers import InternalLayer, layer_modules
from tensorforce.core.networks import LayerbasedNetwork


class LayeredNetwork(LayerbasedNetwork):
    """
    Network consisting of a sequence of layers which can be created from a specification dict.
    """

    # (requires layers as first argument)
    def __init__(self, name, layers, inputs_spec, l2_regularization=None, summary_labels=None):
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
            if 'name' in layers_spec:
                layer_name = layers_spec['name']
            else:
                if isinstance(layers_spec['type'], str):
                    layer_type = layers_spec['type']
                else:
                    layer_type = 'layer'
                layer_name = layer_type + str(layer_counter[layer_type])
                layer_counter[layer_type] += 1

            self.add_module(name=layer_name, module=layers_spec)

    # (requires layers as first argument)
    @classmethod
    def internals_spec(cls, name=None, layers=None, network=None, **kwargs):
        internals_spec = super().internals_spec(network=network)

        if network is None:
            for name, internal_spec in cls.internals_from_layers_spec(
                name=name, layers_spec=layers, layer_counter=Counter()
            ):
                if name in internals_spec:
                    raise TensorforceError.unexpected()
                internals_spec[name] = internal_spec

        return internals_spec

    @classmethod
    def internals_from_layers_spec(cls, name, layers_spec, layer_counter):
        if isinstance(layers_spec, list):
            for spec in layers_spec:
                yield from cls.internals_from_layers_spec(
                    name=name, layers_spec=spec, layer_counter=layer_counter
                )

        else:
            if 'name' in layers_spec:
                layer_name = layers_spec['name']
            else:
                if isinstance(layers_spec['type'], str):
                    layer_type = layers_spec['type']
                else:
                    layer_type = 'layer'
                layer_name = layer_type + str(layer_counter[layer_type])
                layer_counter[layer_type] += 1

            layer_cls, first_arg, kwargs = Module.get_module_class_and_kwargs(
                name=layer_name, module=layers_spec, modules=layer_modules
            )
            if issubclass(layer_cls, InternalLayer):
                if first_arg is None:
                    internals_spec = layer_cls.internals_spec(**kwargs)
                else:
                    internals_spec = layer_cls.internals_spec(first_arg, **kwargs)
                for internal_name, spec in internals_spec.items():
                    internal_name = '{}-{}-{}'.format(name, layer_name, internal_name)
                    yield internal_name, spec

    def tf_apply(self, x, internals, return_internals=False):
        if isinstance(x, dict):
            x = x[next(iter(x))]

        next_internals = OrderedDict()
        for layer in self.modules.values():
            if isinstance(layer, InternalLayer):
                layer_internals = {
                    name: internals['{}-{}-{}'.format(self.name, layer.name, name)]
                    for name in layer.internals_spec()
                }
                assert len(layer_internals) > 0
                x, layer_internals = layer.apply(x=x, **layer_internals)
                for name, internal in layer_internals.items():
                    next_internals['{}-{}-{}'.format(self.name, layer.name, name)] = internal

            else:
                x = layer.apply(x=x)

        if return_internals:
            return x, next_internals
        else:
            return x
