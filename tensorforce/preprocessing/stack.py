# Copyright 2016 reinforce.io. All Rights Reserved.
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

"""
Preprocessing stack class
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class Stack(object):
    def __init__(self):
        self._stack = list()

    def __iadd__(self, other):
        self._stack.append(other)
        return self

    append = __iadd__

    def process(self, state):
        """
        Process state.

        :param state: state array
        :return: new state array
        """
        for processor in self._stack:
            state = processor.process(state)

        return state

    def shape(self, original_shape):
        """
        Return output shape of stack

        :param original_shape: original shape array
        :return: new shape array
        """
        shape = original_shape
        for processor in self._stack:
            shape = processor.shape(shape)

        return shape
