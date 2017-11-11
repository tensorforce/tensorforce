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

from tensorforce.core.networks import Linear, Network
from tensorforce.core.baselines import Baseline


class NetworkBaseline(Baseline):
    """
    Baseline based on a TensorForce network, used when parameters are shared between
    the value function and the baseline.
    """

    def __init__(self, network_spec, scope='network-baseline', summary_labels=()):
        """
        Network baseline.

        Args:
            network_spec: Network specification dict
        """
        self.network = Network.from_spec(
            spec=network_spec,
            kwargs=dict(summary_labels=summary_labels)
        )
        assert len(self.network.internal_inputs()) == 0

        self.linear = Linear(size=1, bias=0.0, scope='prediction')

        super(NetworkBaseline, self).__init__(scope, summary_labels)

    def tf_predict(self, states, update):
        embedding = self.network.apply(x=states, internals=(), update=update)
        prediction = self.linear.apply(x=embedding)
        return tf.squeeze(input=prediction, axis=1)

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the baseline regularization loss.

        Returns:
            Regularization loss tensor
        """
        regularization_loss = super(NetworkBaseline, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.network.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.linear.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        baseline_variables = super(NetworkBaseline, self).get_variables(include_non_trainable=include_non_trainable)
        network_variables = self.network.get_variables(include_non_trainable=include_non_trainable)
        layer_variables = self.linear.get_variables(include_non_trainable=include_non_trainable)

        return baseline_variables + network_variables + layer_variables

    def get_summaries(self):
        baseline_summaries = super(NetworkBaseline, self).get_summaries()
        network_summaries = self.network.get_summaries()
        layer_summaries = self.linear.get_summaries()

        return baseline_summaries + network_summaries + layer_summaries
