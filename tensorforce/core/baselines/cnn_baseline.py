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

from tensorforce.core.baselines import NetworkBaseline


class CNNBaseline(NetworkBaseline):
    """
    CNN baseline (single-state) consisting of convolutional layers followed by dense layers.
    """

    def __init__(self, name, conv_sizes, dense_sizes, l2_regularization=None, summary_labels=None):
        """
        CNN baseline.

        Args:
            conv_sizes: List of convolutional layer sizes
            dense_sizes: List of dense layer sizes
        """
        network_spec = list()

        # Convolution layers
        for size in conv_sizes:
            network_spec.append(dict(type='conv2d', size=size))

        # First layer with larger window
        network_spec[0]['window'] = 5

        # Global max-pooling
        network_spec.append(dict(type='global_pooling', pooling='max'))

        # Dense layers
        for size in dense_sizes:
            network_spec.append(dict(type='dense', size=size))

        super().__init__(
            name=name, network_spec=network_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )
