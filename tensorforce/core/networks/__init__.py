# Copyright 2017 reinforce.io. All Rights Reserved.
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

from tensorforce.core.networks.layer import Layer, Input, Output, TFLayer, Nonlinearity, Dropout, Flatten, Pool2d, Embedding, Linear, Dense, \
    Dueling, Conv1d, Conv2d, InternalLstm, Lstm
from tensorforce.core.networks.network import Network, LayerBasedNetwork, LayeredNetwork


layers = dict(
    input=Input,
    output=Output,
    tf_layer=TFLayer,
    nonlinearity=Nonlinearity,
    dropout=Dropout,
    flatten=Flatten,
    pool2d=Pool2d,
    embedding=Embedding,
    linear=Linear,
    dense=Dense,
    dueling=Dueling,
    conv1d=Conv1d,
    conv2d=Conv2d,
    internal_lstm=InternalLstm,
    lstm=Lstm
)


__all__ = [
    'layers',
    'Layer',
    'Input',
    'Output',
    'TFLayer',
    'Nonlinearity',
    'Dropout',
    'Flatten',
    'Pool2d',
    'Embedding',
    'Linear',
    'Dense',
    'Dueling',
    'Conv1d',
    'Conv2d',
    'InternalLstm',
    'Lstm',
    'Network',
    'LayerBasedNetwork',
    'LayeredNetwork'
]
