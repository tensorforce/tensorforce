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
Linear value function for baseline prediction in TRPO.

N.b. as part of TRPO implementation from https://github.com/ilyasu123/trpo

This code is under MIT license, for more information see LICENSE-EXT.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorforce.core.value_functions import ValueFunction


class LinearValueFunction(ValueFunction):

    def __init__(self, session):
        super(LinearValueFunction, self).__init__(session)
        self.coefficients = None

    def create_tf_operations(self, config):
        if len(config.states) > 1:
            raise Exception()

    def predict(self, states):
        """
        Predict episode value based on linear coefficients.

        :param episode:
        :return: Returns value estimate or 0 if coefficients have not been set
        """
        if self.coefficients is None:
            return np.zeros(shape=(states.shape[0], 1))
        else:
            return self.features(states).dot(self.coefficients)

    def update(self, states, returns):
        features = self.features(states)
        columns = features.shape[1]
        self.coefficients = np.linalg.lstsq(
            features.T.dot(features) + 2 * np.identity(columns), features.T.dot(returns))[0]

    def features(self, states):
        states = next(iter(states.values()))
        states = states.reshape(states.shape[0], -1)
        al = np.arange(states.shape[0]).reshape(-1, 1) / 100.0

        return np.concatenate([states, states ** 2, al, al ** 2, np.ones((states.shape[0], 1))], axis=1)
