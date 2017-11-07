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

from tensorforce.core.preprocessing.preprocessor import Preprocessor
from tensorforce.core.preprocessing.sequence import Sequence
from tensorforce.core.preprocessing.standardize import Standardize
from tensorforce.core.preprocessing.normalize import Normalize
from tensorforce.core.preprocessing.grayscale import Grayscale
from tensorforce.core.preprocessing.image_resize import ImageResize
from tensorforce.core.preprocessing.divide import Divide
from tensorforce.core.preprocessing.clip import Clip
from tensorforce.core.preprocessing.preprocessing import Preprocessing


preprocessors = dict(
    sequence=Sequence,
    standardize=Standardize,
    normalize=Normalize,
    grayscale=Grayscale,
    image_resize=ImageResize,
    divide=Divide,
    clip=Clip,
)


__all__ = [
    'Preprocessor',
    'Sequence',
    'Standardize',
    'Normalize',
    'Grayscale',
    'ImageResize',
    'Preprocessing',
    'Divide',
    'Clip',
    'preprocessors'
]
