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
    Automatically configured layer-based network.
    """

    def __init__(
        self, name, inputs_spec, size=64, depth=2, final_size=None, final_depth=1,
        internal_rnn=False, l2_regularization=None, summary_labels=None
    ):
        super().__init__(
            name=name, inputs_spec=inputs_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

        self.size = size
        self.depth = depth
        self.final_size = size if final_size is None else final_size
        self.final_depth = final_depth
        self.internal_rnn = internal_rnn

        # State-specific layers
        self.state_specific_layers = OrderedDict()
        for name, spec in inputs_spec.items():
            layers = list()

            # Retrieve state
            layers.append(
                self.add_module(name=(name + '-retrieve'), module='retrieve', tensors=name)
            )

            # Embed bool and int states
            if spec['type'] in ('bool', 'int'):
                layers.append(
                    self.add_module(name=(name + '-embedding'), module='embedding', size=self.size)
                )
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
                layers.append(self.add_module(name=(name + '-flatten'), module='flatten'))
                layer = 'dense'
            else:
                raise TensorforceError.unexpected()

            # Repeat layer according to depth (one less if embedded)
            for n in range(self.depth - embedding):
                layers.append(
                    self.add_module(
                        name=(name + '-' + layer + str(n)), module=layer, size=self.size
                    )
                )

            # Max pool if rank greater than one
            if len(spec['shape']) > 1 - embedding:
                layers.append(
                    self.add_module(name=(name + '-pooling'), module='pooling', reduction='max')
                )

            # Register state-specific embedding
            layers.append(
                self.add_module(
                    name=(name + '-register'), module='register',
                    tensor='{}-{}-embedding'.format(self.name, name)
                )
            )

            self.state_specific_layers[name] = layers

        # Final combined layers
        self.final_layers = list()

        # Retrieve state-specific embeddings
        self.final_layers.append(
            self.add_module(
                name='retrieve', module='retrieve',
                tensors=tuple('{}-{}-embedding'.format(self.name, name) for name in inputs_spec),
                aggregation='concat'
            )
        )

        # Repeat layer according to depth
        if len(inputs_spec) > 1:
            for n in range(self.final_depth):
                self.final_layers.append(
                    self.add_module(name=('dense' + str(n)), module='dense', size=self.final_size)
                )

        # Internal Rnn
        if self.internal_rnn:
            self.internal_rnn = self.add_module(
                name='internal_lstm', module='internal_lstm', size=self.final_size
            )
        else:
            self.internal_rnn = None

    @classmethod
    def internals_spec(
        cls, name=None, size=64, final_size=None, internal_rnn=False, network=None, **kwargs
    ):
        internals_spec = super().internals_spec(network=network)

        if network is None and internal_rnn:
            final_size = size if final_size is None else final_size
            for internal_name, spec in InternalLstm.internals_spec(size=final_size).items():
                internals_spec['{}-internal_lstm-{}'.format(name, internal_name)] = spec

        return internals_spec

    def tf_apply(self, x, internals, return_internals=False):
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
                name: internals['{}-internal_lstm-{}'.format(self.name, name)]
                for name in self.internal_rnn.internals_spec()
            }
            assert len(internals) > 0
            tensor, internals = self.internal_rnn.apply(x=tensor, **internals)
            for name, internal in internals.items():
                next_internals['{}-internal_lstm-{}'.format(self.name, name)] = internal

        if return_internals:
            return tensor, next_internals
        else:
            return tensor
