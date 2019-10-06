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

from tensorforce import TensorforceError
from tensorforce.core.layers import InternalLstm
from tensorforce.core.networks import LayerbasedNetwork


class AutoNetwork(LayerbasedNetwork):
    """
    Network which is automatically configured based on its input tensors, offering high-level
    customization (specification key: `auto`).

    Args:
        name (string): Network name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        inputs_spec (specification): Input tensors specification
            (<span style="color:#0000C0"><b>internal use</b></span>).
        size (int > 0): Layer size, before concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: 64).
        depth (int > 0): Number of layers per state, before concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: 2).
        final_size (int > 0): Layer size after concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: layer size).
        final_depth (int > 0): Number of layers after concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: 1).
        internal_rnn (false | parameter, long >= 0): Whether to add an internal state LSTM cell
            as last layer, and if so, horizon of the LSTM
            (<span style="color:#00C000"><b>default</b></span>: false).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(
        self, name, inputs_spec, size=64, depth=2, final_size=None, final_depth=1,
        internal_rnn=False, device=None, summary_labels=None, l2_regularization=None
    ):
        # Some defaults require change in internals_spec
        super().__init__(
            name=name, inputs_spec=inputs_spec, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        self.size = size
        self.depth = depth
        self.final_size = size if final_size is None else final_size
        self.final_depth = final_depth
        self.internal_rnn = internal_rnn

        # State-specific layers
        self.state_specific_layers = OrderedDict()
        # prefix = self.name + '-'
        prefix = ''
        for name, spec in inputs_spec.items():
            layers = list()

            # Retrieve state
            layers.append(self.add_module(
                name=(prefix + name + '-retrieve'), module='retrieve', tensors=name
            ))

            # Embed bool and int states
            if spec['type'] in ('bool', 'int'):
                layers.append(self.add_module(
                    name=(prefix + name + '-embedding'), module='embedding', size=self.size
                ))
                embedding = 1
            else:
                embedding = 0

            # Shape-specific layer type
            if len(spec['shape']) == 1 - embedding:
                layer = 'dense'
            elif len(spec['shape']) == 2 - embedding:
                layer = 'conv1d'
            elif len(spec['shape']) == 3 - embedding:
                layer = 'conv2d'
            elif len(spec['shape']) == 0:
                layers.append(self.add_module(name=(prefix + name + '-flatten'), module='flatten'))
                layer = 'dense'
            else:
                raise TensorforceError.unexpected()

            # Repeat layer according to depth (one less if embedded)
            for n in range(self.depth - embedding):
                layers.append(self.add_module(
                    name=(prefix + name + '-' + layer + str(n)), module=layer, size=self.size
                ))

            # Max pool if rank greater than one
            if len(spec['shape']) > 1 - embedding:
                layers.append(self.add_module(
                    name=(prefix + name + '-pooling'), module='pooling', reduction='max'
                ))

            # Register state-specific embedding
            layers.append(self.add_module(
                name=(prefix + name + '-register'), module='register',
                tensor='{}-{}-embedding'.format(self.name, name)
            ))

            self.state_specific_layers[name] = layers

        # Final combined layers
        self.final_layers = list()

        # Retrieve state-specific embeddings
        self.final_layers.append(self.add_module(
            name=(prefix + 'retrieve'), module='retrieve',
            tensors=tuple('{}-{}-embedding'.format(self.name, name) for name in inputs_spec),
            aggregation='concat'
        ))

        # Repeat layer according to depth
        if len(inputs_spec) > 1:
            for n in range(self.final_depth):
                self.final_layers.append(self.add_module(
                    name=(prefix + 'dense' + str(n)), module='dense', size=self.final_size
                ))

        # Internal Rnn
        if self.internal_rnn is False:
            self.internal_rnn = None
        else:
            self.internal_rnn = self.add_module(
                name=(prefix + 'internal_lstm'), module='internal_lstm', size=self.final_size,
                length=self.internal_rnn
            )

    @classmethod
    def internals_spec(
        cls, network=None, name=None, size=None, final_size=None, internal_rnn=None, **kwargs
    ):
        internals_spec = OrderedDict()

        if network is None:
            assert name is not None
            if size is None:
                size = 64
            if internal_rnn is None:
                internal_rnn = False

            if internal_rnn > 0:
                final_size = size if final_size is None else final_size
                for internal_name, spec in InternalLstm.internals_spec(size=final_size).items():
                    internals_spec[name + '-' + internal_name] = spec

        else:
            assert name is None and size is None and final_size is None and internal_rnn is None

            if network.internal_rnn is not None:
                for internal_name, spec in network.internal_rnn.__class__.internals_spec(
                    layer=network.internal_rnn
                ).items():
                    internals_spec[network.name + '-' + internal_name] = spec

        return internals_spec

    def internals_init(self):
        internals_init = OrderedDict()

        if self.internal_rnn is not None:
            for name, internal_init in self.internal_rnn.internals_init().items():
                internals_init[self.name + '-' + name] = internal_init

        return internals_init

    def tf_apply(self, x, internals, return_internals=False):
        super().tf_apply(x=x, internals=internals, return_internals=return_internals)

        # State-specific layers
        for name, layers in self.state_specific_layers.items():
            tensor = x[name]
            for layer in layers:
                tensor = layer.apply(x=tensor)

        # Final combined layers
        for layer in self.final_layers:
            tensor = layer.apply(x=tensor)

        # Internal Rnn
        next_internals = OrderedDict()
        if self.internal_rnn is not None:
            internals = {
                name: internals[self.name + '-' + name]
                for name in self.internal_rnn.__class__.internals_spec(layer=self.internal_rnn)
            }
            assert len(internals) > 0
            tensor, internals = self.internal_rnn.apply(x=tensor, initial=internals)
            for name, internal in internals.items():
                next_internals[self.name + '-' + name] = internal

        if return_internals:
            return tensor, next_internals
        else:
            return tensor
