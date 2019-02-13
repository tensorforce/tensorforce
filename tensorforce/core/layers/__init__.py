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

from tensorforce.core.layers.layer import Layer, Register, Retrieve, TransformationBase

from tensorforce.core.layers.convolutions import Conv1d, Conv2d
from tensorforce.core.layers.dense import Dense, Linear
from tensorforce.core.layers.embeddings import Embedding
from tensorforce.core.layers.internal_layers import InternalGru, InternalLayer, InternalLstm
from tensorforce.core.layers.keras import Keras
from tensorforce.core.layers.misc import Activation, Dropout
from tensorforce.core.layers.poolings import Flatten, Pooling, Pool1d, Pool2d
from tensorforce.core.layers.rnns import Gru, Lstm, Rnn


layer_modules = dict(
    activation=Activation, conv1d=Conv1d, conv2d=Conv2d, dense=Dense, dropout=Dropout,
    embedding=Embedding, flatten=Flatten, gru=Gru, internal_gru=InternalGru,
    internal_lstm=InternalLstm, keras=Keras, linear=Linear, lstm=Lstm, pooling=Pooling,
    pool1d=Pool1d, pool2d=Pool2d, register=Register, retrieve=Retrieve, rnn=Rnn
)


__all__ = [
    'Activation', 'Conv1d', 'Conv2d', 'Dense', 'Dropout', 'Embedding', 'Flatten', 'GRU',
    'InternalGru', 'InternalLayer', 'InternalLstm', 'Keras', 'Layer', 'layer_modules', 'Linear',
    'Lstm', 'Nonlinearity', 'Pooling', 'Pool1d', 'Pool2d', 'Register', 'Retrieve', 'Rnn',
    'TransformationBase'
]
