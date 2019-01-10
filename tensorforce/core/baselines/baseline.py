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

import tensorflow as tf

from tensorforce.core import Module


class Baseline(Module):
    """
    Base class for baseline value functions.
    """

    def __init__(self, name, inputs_spec, l2_regularization=None, summary_labels=None):
        super().__init__(
            name=name, l2_regularization=l2_regularization, summary_labels=summary_labels
        )

        self.inputs_spec = inputs_spec

    def tf_predict(self, states, internals):
        """
        Creates the TensorFlow operations for predicting the value function of given states.
        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            update: Boolean tensor indicating whether this call happens during an update.
        Returns:
            State value tensor
        """
        raise NotImplementedError

    def tf_reference(self, states, internals, reward):
        """
        Creates the TensorFlow operations for obtaining the reference tensor(s), in case of a
        comparative loss.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Reference tensor(s).
        """
        return None

    def tf_total_loss(self, states, internals, reward):
        """
        Creates the TensorFlow operations for calculating the L2 loss between predicted
        state values and actual rewards.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.
            reference: Optional reference tensor(s), in case of a comparative loss.

        Returns:
            Loss tensor
        """
        prediction = self.predict(states=states, internals=internals)
        return 0.5 * tf.reduce_sum(input_tensor=tf.square(x=(prediction - reward)))
