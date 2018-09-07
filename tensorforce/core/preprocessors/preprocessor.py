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

import tensorflow as tf
import copy

from tensorforce import util
import tensorforce.core.preprocessors


class Preprocessor(object):
    """
    A Preprocessor is an object used to map input state signals to some RL-model
    to some "preprocessed state signals". For example: If the input state is an RGB image of 84x84px (3 color
    channels; 84x84x3 tensor), a preprocessor could make this image a grayscale 84x84x1 tensor, instead.

    Each preprocessor is fully integrated into the model's graph, has its own scope and owns some
    variables that live under that scope in the graph.
    """

    def __init__(self, shape, scope='preprocessor', summary_labels=None):
        self.shape = shape
        self.summary_labels = set(summary_labels or ())
        self.variables = dict()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if registered:
                pass
            elif name in self.variables:
                assert variable is self.variables[name]
            else:
                assert not kwargs['trainable']
                self.variables[name] = variable
            return variable

        self.process = tf.make_template(
            name_=(scope + '/process'),
            func_=self.tf_process,
            custom_getter_=custom_getter
        )
        self.reset = tf.make_template(
            name_=(scope + '/reset'),
            func_=self.tf_reset,
            custom_getter_=custom_getter
        )

    def tf_reset(self):
        """
        Resets this preprocessor to some initial state. This method is called whenever an episode ends.
        This could be useful if the preprocessor stores certain episode-sequence information to do the processing
        and this information has to be reset after the episode terminates.
        """
        pass

    def tf_process(self, tensor):
        """
        Process state (tensor).

        Args:
            tensor (tf.Tensor): The Tensor to process.

        Returns: The pre-processed Tensor.
        """
        return tensor

    def processed_shape(self, shape):
        """
        Shape of preprocessed state given original shape.

        Args:
            shape (tuple): The original (unprocessed) shape.

        Returns: The processed tensor shape.
        """
        return shape

    def get_variables(self):
        """
        Returns the TensorFlow variables used by the preprocessor.

        Returns:
            List of variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]


class PreprocessorStack(object):
    """
    A class to handle many Preprocessor objects applied in a sequence to some state. For example: An image
    tensor as state signal could be re-sized first, then grayscaled, then normalized.
    """

    def __init__(self):
        self.preprocessors = list()

    def reset(self):
        """
        Calls `reset` on all our Preprocessor objects.

        Returns:
            A list of tensors to be fetched.
        """
        fetches = []
        for processor in self.preprocessors:
            fetches.extend(processor.reset() or [])
        return fetches

    def process(self, tensor):
        """
        Process state.

        Args:
            tensor: tensor to process

        Returns: processed state

        """
        for processor in self.preprocessors:
            tensor = processor.process(tensor=tensor)
        return tensor

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

    def get_variables(self):
        return [variable for preprocessor in self.preprocessors for variable in preprocessor.get_variables()]

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a preprocessing stack from a specification dict.
        """
        if isinstance(spec, dict):
            spec = [spec]

        stack = PreprocessorStack()
        for preprocessor_spec in spec:
            # need to deep copy, otherwise will add first processors spec_ to kwargs to second processor
            preprocessor_kwargs = copy.deepcopy(kwargs)
            preprocessor = util.get_object(
                obj=preprocessor_spec,
                predefined_objects=tensorforce.core.preprocessors.preprocessors,
                kwargs=preprocessor_kwargs
            )
            assert isinstance(preprocessor, Preprocessor)
            stack.preprocessors.append(preprocessor)

        return stack
