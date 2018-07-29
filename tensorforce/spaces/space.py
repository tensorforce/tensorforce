class Space(object):
    """
    Defines the observation and action spaces. Used for environment env.states and env.actions.

    Inspired by openai gym's space spaces
    """
    def __init__(self, shape=None, dtype=None):
        import numpy as np
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)

    def sample(self):
        """
        Randomly samples from the space
        """
        raise NotImplementedError

    def contains(self, point):
        """
        Returns the bool value of point being contained in the action space

        Params
        ------
            point : a point that has the same shape and dtype as this space.
        """
        raise NotImplementedError

    @property
    def discrete(self):
        """
        Returns the bool of whether the space is discrete
        """
        raise NotImplementedError
