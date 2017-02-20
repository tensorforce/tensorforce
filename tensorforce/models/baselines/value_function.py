# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Generic baseline value function for policy gradient methods.
"""

import numpy as np


class ValueFunction(object):

    def get_features(self, path):
        states = path["states"]
        states = states.reshape(states.shape[0], -1)

        path_length = len(path["rewards"])
        al = np.arange(path_length).reshape(-1, 1) / 100.0

        return np.concatenate([states, states ** 2, al, al ** 2, np.ones((path_length, 1))], axis=1)

    def predict(self, path):
        raise NotImplementedError

    def fit(self, paths):
        raise NotImplementedError