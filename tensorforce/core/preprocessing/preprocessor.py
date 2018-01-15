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


class Preprocessor(object):

    def __init__(self, scope='preprocessor', summary_labels=None):
        self.summary_labels = set(summary_labels or ())
        self.variables = dict()
        self.summaries = list()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if not registered:
                self.variables[name] = variable
            return variable

        self.process = tf.make_template(
            name_=(scope + '/process'),
            func_=self.tf_process,
            custom_getter_=custom_getter
        )

    def reset(self):
        pass

    def tf_process(self, tensor):
        """
        Process state.

        Args:
            tensor: tensor to process.

        Returns: processed tensor.
        """
        return tensor

    def processed_shape(self, shape):
        """
        Shape of preprocessed state given original shape.

        Args:
            shape: original shape.

        Returns: processed tensor shape
        """
        return shape

    def get_variables(self):
        """
        Returns the TensorFlow variables used by the preprocessor.

        Returns:
            List of variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]
