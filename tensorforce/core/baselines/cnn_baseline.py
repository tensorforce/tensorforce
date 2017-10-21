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

from tensorforce.core.baselines import NetworkBaseline


class CNNBaseline(NetworkBaseline):
    """
    CNN baseline (single-state) consisting of convolutional layers followed by dense layers.
    """

    def __init__(self, conv_sizes, dense_sizes, scope='cnn-baseline', summary_labels=()):
        """
        CNN baseline.

        Args:
            conv_sizes: List of convolutional layer sizes
            dense_sizes: List of dense layer sizes
        """

        layers_spec = []
        for size in conv_sizes:
            layers_spec.append({'type': 'conv2d', 'size': size, 'stride': 1, 'window': 3})

        # First layer has a larger window by convention.
        layers_spec[0]['window'] = 5

        layers_spec.append({'type': 'flatten'})  # TODO: change to max pooling!
        for size in dense_sizes:
            layers_spec.append({'type': 'dense', 'size': size})

        super(CNNBaseline, self).__init__(layers_spec, scope, summary_labels)
