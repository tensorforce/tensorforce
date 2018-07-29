import numpy as np
import tensorforce

from tensorforce.spaces.space import Space

class DiscreteBox(Space):
    """
    A box in R^n where each point is discrete
    """
    def __init__(self, low=None, high=None, shape=None, dtype=int):
        """
        Initializes in two ways

        1) low and high are int scalars where low < high. This creates a box of shape
        {@arg shape} with lower and upper bounds low and high.

        2) low and high are np.int of the same shape. Creates a box with lower and
        upper bounds for each value.

        Params
        ------
            low (int or np.int):
                the lower bound values in the box
            high (int or np.int):
                the upper bound values in the box
            shape (tuple of int):
                the shape (used when low, high are ints)
            dtype (type):
                used only for base class, not to be changed

        Example Usage
        -------------
            DiscreteBox(low=-1, high=1, shape=(5, 5)) intializes a box with [low, high]
                in shape dimensions
            DiscreteBox(low=np.array([-5, -6]), high=np.array([5, 6])) initializes a box
                with [low, high] 

        Raises
        ------
            ValueError('dtype should be int') if dtype is not int
            ValueError('low should be less than high') if low >= high somewhere
            ValueError('low should be same shape as high') if low and high are
                np arrays with different shape
            ValueError('low and high should not be nones') if low and high are
                not Nones
        """
        if dtype is not int:
            raise ValueError('dtype should be int')

        if low is None or high is None:
            raise ValueError('low and high should not be nones')

        if np.isscalar(low) and np.isscalar(high):
            if low > high:
                raise ValueError('low should be less than high')
            self.low = int(low) * np.ones(shape)
            self.high = int(high) * np.ones(shape)
        else:
            if low.shape != high.shape:
                raise ValueError('low should be the same shape as high')
            diff = low <= high
            if not np.any(diff):
                raise ValueError('low should be less than high')
            self.low = low
            self.high = high
        if shape is None:
            tensorforce.spaces.Space.__init__(low.shape, dtype)
        else:
            tensorforce.spaces.Space.__init__(shape, dtype)

    def sample(self):
        """
        Randomly samples from the space. In this case an integer
        np array of same shape as self.low and self.high
        """
        difference = self.high - self.low
        sample_ = (np.random.uniform(size=self.high.shape) * difference).astype(np.int64)
        return self.low + sample_

    def contains(self, point):
        """
        Returns the bool value of {@ param point} being contained in the action space

        Params
        ------
            point (np.ndarray that is np.int64):
                a point that has the same shape and dtype as this space.
        """
        higher = np.all(point >= self.low)
        lower = np.all(point <= self.high)
        return higher and lower

    @property
    def discrete(self):
        """
        Returns the bool of whether the space is discrete
        """
        return True
