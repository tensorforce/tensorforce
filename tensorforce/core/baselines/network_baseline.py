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
    Baseline based on a TensorForce network
    """

    def __init__(self, network_spec, scope='network-baseline', summary_level=0):
        """
        Network baseline

        Args:
            network_spec: Network specification dict
        """
        with tf.name_scope(name=scope):
            self.network = Network.from_spec(spec=network_spec)
            assert len(self.network.internal_inputs()) == 0

            self.linear = Linear(size=1, bias=0.0, scope='prediction')

        super(NetworkBaseline, self).__init__(scope, summary_level)

    def tf_predict(self, states):
        embedding = self.network.apply(x=states)
        prediction = self.linear.apply(x=embedding)
        return tf.squeeze(input=prediction, axis=1)

    def get_variables(self):
        return [self.variables[key] for key in sorted(self.variables)] + self.network.get_variables() + self.linear.get_variables()
