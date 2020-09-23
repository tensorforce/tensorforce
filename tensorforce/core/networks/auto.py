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

from tensorforce import TensorforceError
from tensorforce.core.networks import LayeredNetwork


class AutoNetwork(LayeredNetwork):
    """
    Network which is automatically configured based on its input tensors, offering high-level
    customization (specification key: `auto`).

    Args:
        size (int > 0): Layer size, before concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: 64).
        depth (int > 0): Number of layers per state, before concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: 2).
        final_size (int > 0): Layer size after concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: layer size).
        final_depth (int > 0): Number of layers after concatenation if multiple states
            (<span style="color:#00C000"><b>default</b></span>: 1).
        rnn (false | [parameter](../modules/parameters.html), int >= 0): Whether to add an LSTM cell
            with internal state as last layer, and if so, horizon of the LSTM for truncated
            backpropagation through time
            (<span style="color:#00C000"><b>default</b></span>: false).
        device (string): Device name
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        l2_regularization (float >= 0.0): Scalar controlling L2 regularization
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        name (string): <span style="color:#0000C0"><b>internal use</b></span>.
        inputs_spec (specification): <span style="color:#0000C0"><b>internal use</b></span>.
    """

    def __init__(
        self, *, size=64, depth=2, final_size=None, final_depth=1, rnn=False, device=None,
        l2_regularization=None, name=None, inputs_spec=None,
        # Deprecated
        internal_rnn=None
    ):
        if internal_rnn is not None:
            raise TensorforceError.deprecated(
                name='AutoNetwork', argument='internal_rnn', replacement='rnn'
            )

        if len(inputs_spec) == 1:
            if final_size is not None:
                raise TensorforceError.invalid(
                    name='AutoNetwork', argument='final_size', condition='input size = 1'
                )
            if final_depth is not None and final_depth != 1:
                raise TensorforceError.invalid(
                    name='AutoNetwork', argument='final_depth', condition='input size = 1'
                )

        if final_size is None:
            final_size = size
        if final_depth is None:
            final_depth = 0

        layers = list()
        for input_name, spec in inputs_spec.items():
            state_layers = list()
            layers.append(state_layers)

            # Retrieve input state
            if input_name is None:
                prefix = ''
            else:
                prefix = input_name.replace('/', '_') + '_'
                state_layers.append(dict(
                    type='retrieve', name=(prefix + 'retrieve'), tensors=(input_name,)
                ))

            # Embed bool and int states
            requires_embedding = (spec.type == 'bool' or spec.type == 'int')
            if spec.type == 'int' and spec.num_values is None:
                if input_name is None:
                    raise TensorforceError.required(
                        name='state', argument='num_values', condition='state type is int'
                    )
                else:
                    raise TensorforceError.required(
                        name=(input_name + ' state'), argument='num_values',
                        condition='state type is int'
                    )
            if requires_embedding:
                state_layers.append(dict(
                    type='embedding', name=(prefix + 'embedding'), size=size
                ))

            # Shape-specific layer type
            if spec.rank == 1 - requires_embedding:
                layer = 'dense'
            elif spec.rank == 2 - requires_embedding:
                layer = 'conv1d'
            elif spec.rank == 3 - requires_embedding:
                layer = 'conv2d'
            elif spec.rank == 0:
                state_layers.append(dict(type='flatten', name=(prefix + 'flatten')))
                layer = 'dense'
            else:
                raise TensorforceError.value(
                    name='AutoNetwork', argument='input rank', value=spec.rank, hint='>= 3'
                )

            # Repeat layer according to depth (one less if embedded)
            for n in range(depth - requires_embedding):
                state_layers.append(dict(
                    type=layer, name='{}{}{}'.format(prefix, layer, n), size=size
                ))

            # Max pool if rank greater than one
            if spec.rank > 1 - requires_embedding:
                state_layers.append(dict(
                    type='pooling', name=(prefix + 'pooling'), reduction='max'
                ))

            # Register state-specific embedding
            if input_name is not None:
                state_layers.append(dict(
                    type='register', name=(prefix + 'register'), tensor=(input_name + '-embedding')
                ))

        # Final combined layers
        final_layers = list()
        layers.append(final_layers)

        if len(inputs_spec) > 1:
            # Retrieve state-specific embeddings
            final_layers.append(dict(
                type='retrieve', name='retrieve',
                tensors=tuple(input_name + '-embedding' for input_name in inputs_spec),
                aggregation='concat'
            ))

            # Repeat layer according to depth
            for n in range(final_depth):
                final_layers.append(dict(type='dense', name=('dense' + str(n)), size=final_size))

        # Rnn
        if rnn is not None and rnn is not False:
            final_layers.append(dict(type='lstm', name='lstm', size=final_size, horizon=rnn))

        super().__init__(
            layers=layers, device=device, l2_regularization=l2_regularization, name=name,
            inputs_spec=inputs_spec
        )
