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

# from tensorforce.core.layers.layer import TFLayer, Flatten, Pool2d, Dueling, GlobalPooling

from tensorforce.core.layers.convolutions import Conv1d, Conv2d
from tensorforce.core.layers.dense import Dense, Linear
from tensorforce.core.layers.embeddings import Embedding
from tensorforce.core.layers.internal_rnns import InternalGru, InternalLstm
from tensorforce.core.layers.misc import Activation, Dropout
from tensorforce.core.layers.poolings import Pooling
from tensorforce.core.layers.rnns import Gru, Lstm


layer_modules = dict(
    activation=Activation, conv1d=Conv1d, conv2d=Conv2d, dense=Dense, dropout=Dropout,
    embedding=Embedding, gru=Gru, internal_gru=InternalGru, internal_lstm=InternalLstm,
    linear=Linear, lstm=Lstm, pooling=Pooling, register=Register, retrieve=Retrieve
)


__all__ = [
    'Activation', 'Conv1d', 'Conv2d', 'Dense', 'Dropout', 'Embedding', 'GRU', 'InternalGru',
    'InternalLstm', 'Layer', 'layer_modules', 'Linear', 'Lstm', 'Nonlinearity', 'Pooling',
    'Register', 'Retrieve', 'TransformationBase'
]
