# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Preprocessor testing.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from six.moves import xrange

from tensorforce import preprocessing

def test_preprocessing_grayscale():
    """
    Testing grayscale preprocessor. Verifies expected and calculated state shapes.
    """
    pp = preprocessing.grayscale.Grayscale()

    shape = list(np.random.randint(1, 20, size=2)) + [3]
    state = np.random.randint(0, 255, size=shape)

    # verify expected shape
    processed_shape = pp.shape(shape)
    assert tuple(processed_shape) == tuple(shape[0:2])

    # verify calculated shape
    processed_state = pp.process(state)
    assert processed_state.shape == tuple(processed_shape)


def test_preprocessing_concat():
    """
    Testing concat preprocessor. Verifies expected and calculated state shapes.
    """
    concat_length = np.random.randint(1, 10)

    pp = preprocessing.concat.Concat(concat_length)

    shape = list(np.random.randint(1, 20, size=3))
    state = np.random.randint(0, 255, size=shape)

    # verify expected shape
    processed_shape = pp.shape(shape)
    assert tuple(processed_shape) == tuple([concat_length] + shape)

    # verify calculated shape
    processed_state = pp.process(state)
    assert processed_state.shape == tuple(processed_shape)

    # verify calculated content
    states = [state]
    for i in xrange(concat_length-1):
        new_state = np.random.randint(0, 255, size=shape)
        states.append(new_state)
        processed_state = pp.process(new_state)

    assert not (np.array(states) - processed_state).any()

    # add another state
    new_state = np.random.randint(0, 255, size=shape)
    states.append(new_state)
    processed_state = pp.process(new_state)

    assert not (np.array(states[1:]) - processed_state).any()



def test_preprocessing_imresize():
    """
    Testing imresize preprocessor. Verifies expected and calculated state shapes.
    """
    dimensions = list(np.random.randint(10, 20, size=2))

    pp = preprocessing.imresize.Imresize(*dimensions)

    shape = list(np.random.randint(1, 20, size=2))
    state = np.random.randint(0, 255, size=shape)

    # verify expected shape
    processed_shape = pp.shape(shape)
    assert tuple(processed_shape) == tuple(dimensions)

    # verify calculated shape
    processed_state = pp.process(state)
    assert processed_state.shape == tuple(processed_shape)


def test_preprocessing_maximum():
    """
    Testing maximum preprocessor. Verifies expected and calculated state shapes.
    """
    count = np.random.randint(1, 10)

    pp = preprocessing.maximum.Maximum(count)

    shape = list(np.random.randint(1, 20, size=3))
    state = np.random.randint(0, 255, size=shape)

    # verify expected shape
    processed_shape = pp.shape(shape)
    assert tuple(processed_shape) == tuple(shape)

    # verify calculated shape
    processed_state = pp.process(state)
    assert processed_state.shape == tuple(processed_shape)

    # verify calculated content
    states = [state]
    max_state = state.reshape(-1)
    for i in xrange(count - 1):
        new_state = np.random.randint(0, 255, size=shape)

        new_state_reshaped = new_state.reshape(-1)

        # find maximum values manually
        for j, val in enumerate(new_state_reshaped):
            if max_state[j] < val:
                max_state[j] = val

        states.append(new_state)
        processed_state = pp.process(new_state)

    assert not (max_state.reshape(shape) - processed_state).any()
