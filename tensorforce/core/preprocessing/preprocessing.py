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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce import util
from tensorforce.core.preprocessing import Preprocessor
import tensorforce.core.preprocessing


class Preprocessing(object):

    def __init__(self):
        self.preprocessors = list()

    def add(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def process(self, state):
        """
        Process state.

        Args:
            state: state

        Returns: processed state

        """
        for processor in self.preprocessors:
            state = processor.process(state=state)
        return state

    def processed_shape(self, shape):
        """
        Shape of preprocessed state given original shape.

        Args:
            shape: original state shape

        Returns: processed state shape
        """
        for processor in self.preprocessors:
            shape = processor.processed_shape(shape=shape)
        return shape

    def reset(self):
        for processor in self.preprocessors:
            processor.reset()

    @staticmethod
    def from_spec(spec):
        """
        Creates a preprocessing stack from a specification dict.
        """
        if not isinstance(spec, list):
            spec = [spec]

        preprocessing = Preprocessing()
        for spec in spec:
            preprocessor = util.get_object(
                obj=spec,
                predefined_objects=tensorforce.core.preprocessing.preprocessors
            )
            assert isinstance(preprocessor, Preprocessor)
            preprocessing.add(preprocessor=preprocessor)
        return preprocessing
