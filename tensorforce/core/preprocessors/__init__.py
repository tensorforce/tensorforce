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

from tensorforce.core.preprocessors.preprocessor import Preprocessor, PreprocessorStack
from tensorforce.core.preprocessors.sequence import Sequence
from tensorforce.core.preprocessors.standardize import Standardize
from tensorforce.core.preprocessors.running_standardize import RunningStandardize
from tensorforce.core.preprocessors.normalize import Normalize
from tensorforce.core.preprocessors.grayscale import Grayscale
from tensorforce.core.preprocessors.image_resize import ImageResize
from tensorforce.core.preprocessors.divide import Divide
from tensorforce.core.preprocessors.clip import Clip
from tensorforce.core.preprocessors.flatten import Flatten
from tensorforce.core.preprocessors.expand_dims import ExpandDims


preprocessors = dict(
    sequence=Sequence,
    standardize=Standardize,
    running_standardize=RunningStandardize,
    normalize=Normalize,
    grayscale=Grayscale,
    image_resize=ImageResize,
    divide=Divide,
    clip=Clip,
    flatten=Flatten,
    expand_dims=ExpandDims
)


__all__ = [
    'preprocessors',
    'Preprocessor',
    'PreprocessorStack',
    'Sequence',
    'Standardize',
    'RunningStandardize',
    'Normalize',
    'Grayscale',
    'ImageResize',
    'Divide',
    'Clip',
    'Flatten',
    'ExpandDims'
]
