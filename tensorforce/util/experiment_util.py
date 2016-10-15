import numpy as np


def global_seed():
    """
    Convenience function to control random seeding throughout the framework.
    :return: A numpy random number generator with a fixed seed.
    """
    return np.random.RandomState(42)
