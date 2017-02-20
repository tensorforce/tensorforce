# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Linear value function for baseline prediction in TRPO.

N.b. as part of TRPO implementation from https://github.com/ilyasu123/trpo
"""
import numpy as np

from tensorforce.models.baselines.value_function import ValueFunction


class LinearValueFunction(ValueFunction):
    def __init__(self):
        self.coefficients = None


    def fit(self, paths):
        feature_matrix = np.concatenate([self.get_features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        columns = feature_matrix.shape[1]
        lamb = 2

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

