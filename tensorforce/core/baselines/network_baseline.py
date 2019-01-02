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

from tensorforce.core import layer_modules, network_modules
from tensorforce.core.baselines import Baseline


class NetworkBaseline(Baseline):
    """
    Baseline based on a TensorForce network, used when parameters are shared between the value
    function and the baseline.
    """

    def __init__(self, name, network_spec, l2_regularization=None, summary_labels=None):
        """
        Network baseline.

        Args:
            network_spec: Network specification dict
        """
        self.network = self.add_module(
            name='network', module=network_spec, modules=network_modules, default_module='layered'
        )
        assert len(self.network.internals_spec()) == 0

        self.prediction = self.add_module(
            name='prediction', module='linear', modules=layer_modules, size=0
        )

        super().__init__(
            name=name, l2_regularization=l2_regularization, summary_labels=summary_labels
        )

    def tf_predict(self, states, internals, update):
        embedding = self.network.apply(x=states, internals=internals, update=update)

        return self.prediction.apply(x=embedding)
