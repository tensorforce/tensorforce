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

"""
Preprocessing stack class
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.exception import TensorForceError
from tensorforce.core.preprocessing import preprocessors


class Stack(object):
    def __init__(self):
        self._stack = list()

    def __iadd__(self, other):
        self._stack.append(other)
        return self

    append = __iadd__

    def process(self, state):
        """Process state

        Args:
            state: State to process.

        Returns: Processed state.

        """
        for processor in self._stack:
            state = processor.process(state)

        return state

    def shape(self, original_shape):
        """Return output shape of state

        Args:
            original_shape: tuple containing original state

        Returns: tuple containing processed state shape

        """
        shape = original_shape
        for processor in self._stack:
            shape = processor.shape(shape)

        return shape


class MultiStack(object):
    def __init__(self):
        self._state_stacks = dict()

    def add(self, state_name, state_stack):
        """Add state stack to MultiStack collection.

        Args:
            state_name: state name
            state_stack: preprocessing ``Stack``

        Returns: self

        """
        self._state_stacks.update({state_name: state_stack})
        return self

    def process(self, states):
        """Process state dict.

        Args:
            states: dict containing states.

        Returns: dict containing processed states.
        """
        processed_states = dict()
        for state_name, state in states.items():
            stack = self._state_stacks.get(state_name)

            if not stack:
                processed_state = state
            else:
                processed_state = stack.process(state)

            processed_states.update({state_name: processed_state})

        return processed_states

    def shape(self, original_shapes):
        """Return shapes of processed states.

        Args:
            original_shapes: dict containing state shapes.

        Returns: dict containing processed state shapes.
        """
        processed_shapes = dict()
        for state_name, shape in original_shapes.items():
            stack = self._state_stacks.get(state_name)

            if not stack:
                processed_shape = shape
            else:
                processed_shape = stack.shape(shape)

            processed_shapes.update({state_name: processed_shape})

        return processed_shapes


def build_preprocessing_stack(config):
    """Utility function to generate a stack of preprocessors from a config.
    Args:
        config: Either a list containing other lists that describe the preprocessors to stack, e.g.
            [
                ["grayscale"],
                ["imresize", 84, 84],
                ["concat", 4, "append"],
                ["normalize"]
            ]
            or a dict containing items that point to such lists for multi-state preprocessing, e.g.
            {
                "state1": [
                    ["grayscale"],
                    ["imresize", 84, 84],
                    ["concat", 4, "append"],
                    ["normalize"]
                ],
                "state2": [
                    ["maximum", 2],
                    ["standardize"]
                ]
            }

    Returns: preprocessing ``Stack`` or ``MultiStack``.

    """

    if isinstance(config, dict):
        stack = MultiStack()

        for state_name, state_config in config.items():
            state_stack = build_preprocessing_stack(state_config)
            stack.add(state_name, state_stack)

    else:
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
