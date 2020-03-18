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
from tensorforce.core import Module, tf_function
from tensorforce.core.layers import Layer, layer_modules, TemporalLayer
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

        self.inputs_spec = inputs_spec

    def input_signature(self, function):
        if function == 'apply':
            return [
                util.to_tensor_spec(value_spec=self.inputs_spec, batched=True),
                util.to_tensor_spec(
                    value_spec=self.__class__.internals_spec(network=self), batched=True
                )
            ]

        elif function == 'past_horizon':
            return ()

        else:
            return super().input_signature(function=function)

    def get_output_spec(self):
        raise NotImplementedError

    @classmethod
    def internals_spec(cls, network=None, **kwargs):
        return OrderedDict()

    def internals_init(self):
        return OrderedDict()

    def max_past_horizon(self, on_policy):
        raise NotImplementedError

    @tf_function(num_args=0)
    def past_horizon(self, on_policy):
        raise NotImplementedError

    @tf_function(num_args=2)
    def apply(self, x, internals, return_internals):
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
            for layer in network.this_submodules:
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

        for layer in self.this_submodules:
            if not isinstance(layer, StatefulLayer):
                continue
            for name, internal_init in layer.internals_init().items():
                internals_init['{}-{}-{}'.format(self.name, layer.name, name)] = internal_init

        return internals_init

    def add_module(self, *args, **kwargs):
        # Default modules set: layer_modules
        if len(args) < 3 and 'modules' not in kwargs:
            kwargs['modules'] = layer_modules

        if kwargs.get('input_spec') is None:
            if self.output_spec is None:
                if util.is_atomic_values_spec(values_spec=self.inputs_spec):
                    self.output_spec = self.inputs_spec
                # elif len(self.inputs_spec) == 1:
                #     self.output_spec = next(iter(self.inputs_spec.values()))
                else:
                    self.output_spec = next(iter(self.inputs_spec.values()))

            if kwargs.get('input_spec') is None:
                kwargs['input_spec'] = self.output_spec
            else:
                kwargs['input_spec'] = util.unify_value_specs(
                    value_spec1=kwargs['input_spec'], value_spec2=self.output_spec
                )

            layer = super().add_module(*args, **kwargs)

            self.output_spec = layer.output_spec

        else:
            layer = super().add_module(*args, **kwargs)
            self.output_spec = layer.output_spec

        if not isinstance(layer, (Layer, Parameter)):
            raise TensorforceError.type(
                name='layer-based network', argument='sub-module', value=layer
            )

        return layer

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
