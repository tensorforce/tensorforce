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

from tensorforce.preprocessing.stack import Stack
from tensorforce.preprocessing.concat import Concat
from tensorforce.preprocessing.grayscale import Grayscale
from tensorforce.preprocessing.imresize import Imresize
from tensorforce.preprocessing.maximum import Maximum

from tensorforce.preprocessing.normalize import Normalize
from tensorforce.preprocessing.standardize import Standardize
from tensorforce.preprocessing.stack import Stack
from tensorforce.exception import TensorForceError

preprocessors = dict(
    concat=Concat,
    grayscale=Grayscale,
    imresize=Imresize,
    maximum=Maximum,
    normalize=Normalize,
    standardize=Standardize
)


def build_preprocessing_stack(config):
    """Utility function to generate a stack of preprocessors from a config.
    Args:
        config: 

    Returns:

    """
    stack = Stack()

    for preprocessor_conf in config:
        preprocessor_name = preprocessor_conf[0]

        preprocessor_params = []
        if len(preprocessor_conf) > 1:
            preprocessor_params = preprocessor_conf[1:]

        preprocessor_class = preprocessors.get(preprocessor_name, None)

        if not preprocessor_class:
            raise TensorForceError("No such preprocessor: {}".format(preprocessor_name))

        preprocessor = preprocessor_class(*preprocessor_params)
        stack += preprocessor

    return stack

__all__ = ["Stack", "Concat", "Grayscale", "Imresize", "Maximum", "Normalize", "Standardize"]
