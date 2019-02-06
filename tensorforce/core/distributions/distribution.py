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

from tensorforce.core import Module


class Distribution(Module):
    """
    Base class for policy distributions.
    """

    def __init__(self, name, action_spec, embedding_size, summary_labels=None):
        """
        Distribution.

        Args:
            action_spec: Action specification.
        """
        super().__init__(name=name, l2_regularization=0.0, summary_labels=summary_labels)

        self.action_spec = action_spec
        self.embedding_size = embedding_size

    def tf_parametrize(self, x):
        """
        Creates the tensorFlow operations for parametrizing a distribution conditioned on the
        given input.

        Args:
            x: Input tensor which the distribution is conditioned on.

        Returns:
            tuple of distribution parameter tensors.
        """
        raise NotImplementedError

    def tf_sample(self, parameters, deterministic):
        """
        Creates the tensorFlow operations for sampling an action based on a distribution.

        Args:
            parameters: tuple of distribution parameter tensors.
            deterministic: Boolean input tensor indicating whether the maximum likelihood action
                should be returned.

        Returns:
            Sampled action tensor.
        """
        raise NotImplementedError

    def tf_log_probability(self, parameters, action):
        """
        Creates the tensorFlow operations for calculating the log probability of an action for a  
        distribution.

        Args:
            parameters: tuple of distribution parameter tensors.
            action: Action tensor.

        Returns:
            KL divergence tensor.
        """
        raise NotImplementedError

    def tf_entropy(self, parameters):
        """
        Creates the tensorFlow operations for calculating the entropy of a distribution.

        Args:
            parameters: tuple of distribution parameter tensors.

        Returns:
            Entropy tensor.
        """
        raise NotImplementedError

    def tf_kl_divergence(self, parameters1, parameters2):
        """
        Creates the tensorFlow operations for calculating the KL divergence between two  
        distributions.

        Args:
            parameters1: tuple of parameter tensors for first distribution.
            parameters2: tuple of parameter tensors for second distribution.

        Returns:
            KL divergence tensor.
        """
        raise NotImplementedError

    def tf_action_value(self, parameters, action=None):
        raise NotImplementedError

    def tf_states_value(self, parameters):
        raise NotImplementedError
