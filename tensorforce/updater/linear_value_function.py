# Copyright 2016 reinforce.io. All Rights Reserved.
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
"""
import numpy as np


class LinearValueFunction(object):
    def __init__(self):
        self.coefficients = None

    def get_features(self, path):
        states = path["states"]
        states = states.reshape(states.shape[0], -1)

        path_length = len(path["rewards"])
        al = np.arange(path_length).reshape(-1, 1) / 100.0

        return np.concatenate([states, states ** 2, al, al ** 2, np.ones((path_length, 1))], axis=1)

    def fit(self, paths):
        feature_matrix = np.concatenate([self.get_features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        columns = feature_matrix.shape[1]
        lamb = 2.0

        self.coefficients = np.linalg.lstsq(feature_matrix.T.dot(feature_matrix)
                                            + lamb * np.identity(columns), feature_matrix.T.dot(returns))[0]

    def predict(self, path):
        """
        Predict path value based on linear coefficients.

        :param path:
        :return: Returns value estimate or 0 if coefficients have not been set
        """

        if self.coefficients is None:
            return np.zeros(len(path["rewards"]))
        else:
            return self.get_features(path).dot(self.coefficients)

