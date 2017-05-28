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

from tensorforce.core.preprocessing.concat import Concat
from tensorforce.core.preprocessing.imresize import Imresize
from tensorforce.core.preprocessing.maximum import Maximum
from tensorforce.core.preprocessing.normalize import Normalize
from tensorforce.core.preprocessing.standardize import Standardize

from tensorforce.core.preprocessing.grayscale import Grayscale

preprocessors = dict(
    concat=Concat,
    grayscale=Grayscale,
    imresize=Imresize,
    maximum=Maximum,
    normalize=Normalize,
    standardize=Standardize
)

from tensorforce.core.preprocessing.stack import Stack, MultiStack, build_preprocessing_stack

__all__ = ["preprocessors", "Stack", "MultiStack", "build_preprocessing_stack", "Concat", "Grayscale", "Imresize", "Maximum", "Normalize", "Standardize"]
