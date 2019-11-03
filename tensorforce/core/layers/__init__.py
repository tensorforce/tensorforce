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

from tensorforce.core.layers.layer import Layer, StatefulLayer, TemporalLayer, TransformationBase

from tensorforce.core.layers.convolution import Conv1d, Conv1dTranspose, Conv2d, Conv2dTranspose
from tensorforce.core.layers.dense import Dense
from tensorforce.core.layers.embedding import Embedding
from tensorforce.core.layers.internal_rnn import InternalGru, InternalLstm, InternalRnn
from tensorforce.core.layers.keras import Keras
from tensorforce.core.layers.linear import Linear
from tensorforce.core.layers.misc import Activation, Block, Dropout, Function, Register, \
    Retrieve, Reuse
from tensorforce.core.layers.normalization import ExponentialNormalization, InstanceNormalization
from tensorforce.core.layers.pooling import Flatten, Pooling, Pool1d, Pool2d
from tensorforce.core.layers.preprocessing import Clipping, Deltafier, Image, PreprocessingLayer, \
    Sequence
from tensorforce.core.layers.rnn import Gru, Lstm, Rnn


layer_modules = dict(
    activation=Activation, block=Block, clipping=Clipping, conv1d=Conv1d,
    conv1d_transpose=Conv1dTranspose, conv2d=Conv2d, conv2d_transpose=Conv2dTranspose,
    default=Function, deltafier=Deltafier, dense=Dense, dropout=Dropout, embedding=Embedding,
    exponential_normalization=ExponentialNormalization, flatten=Flatten, function=Function,
    gru=Gru, image=Image, instance_normalization=InstanceNormalization, internal_gru=InternalGru,
    internal_lstm=InternalLstm, internal_rnn=InternalRnn, keras=Keras, linear=Linear, lstm=Lstm,
    pooling=Pooling, pool1d=Pool1d, pool2d=Pool2d, register=Register, retrieve=Retrieve,
    reuse=Reuse, rnn=Rnn, sequence=Sequence
)


__all__ = [
    'Activation', 'Block', 'Clipping', 'Conv1d', 'Conv1dTranspose', 'Conv2d', 'Conv2dTranspose',
    'Deltafier', 'Dense', 'Dropout', 'Embedding', 'ExponentialNormalization', 'Flatten',
    'Function', 'GRU', 'Image', 'InstanceNormalization', 'InternalGru', 'InternalLayer',
    'InternalLstm', 'InternalRnn', 'Keras', 'Layer', 'layer_modules', 'Linear', 'Lstm',
    'Nonlinearity', 'Pooling', 'Pool1d', 'Pool2d', 'PreprocessingLayer', 'Register', 'Retrieve',
    'Rnn', 'Sequence', 'StatefulLayer', 'TemporalLayer', 'TransformationBase'
]
