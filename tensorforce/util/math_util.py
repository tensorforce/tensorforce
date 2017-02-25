# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Contains various mathematical utility functions used for policy gradient methods.
"""

import numpy as np
import tensorflow as tf
import scipy.signal


def zero_mean_unit_variance(data):
    """
    Transform array to zero mean unit variance.
    :param data:
    :return:
    """
    data -= data.mean()
    data /= (data.std() + 1e-8)

    return data

def unity_based_normalization(data):
    """
    Transform array to values between [0; 1]
    :param data:
    :return:
    """
    data -= data.min()
    data /= (data.max() - data.min())

    return data


def discount(rewards, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]


def get_shape(variable):
    shape = [k.value for k in variable.get_shape()]
    return shape


def get_number_of_elements(x):
    return np.prod(get_shape(x))


def get_flattened_gradient(loss, variables):
    """
    Flat representation of gradients.

    :param loss: Loss expression
    :param variables: Variables to compute gradients on
    :return: Flattened gradient expressions
    """
    gradients = tf.gradients(loss, variables)

    return tf.concat(axis=0, values=[tf.reshape(grad, [get_number_of_elements(v)])
                         for (v, grad) in zip(variables, gradients)])


class FlatVarHelper(object):
    def __init__(self, session, variables):
        self.session = session
        shapes = map(get_shape, variables)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []

        for (shape, variable) in zip(shapes, variables):
            size = np.prod(shape)
            assigns.append(tf.assign(variable, tf.reshape(self.theta[start:start + size], shape)))
            start += size

        self.set_op = tf.group(*assigns)
        self.get_op = tf.concat(axis=0, values=[tf.reshape(variable, [get_number_of_elements(variable)]) for variable in variables])

    def set(self, theta):
        """
        Assign flat variable representation.

        :param theta: values
        """
        self.session.run(self.set_op, feed_dict={self.theta: theta})

    def get(self):
        """
        Get flat representation.

        :return: Concatenation of variables
        """

        return self.session.run(self.get_op)


def line_search(f, initial_x, full_step, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    """
    Line search for TRPO where a full step is taken first and then backtracked to
    find optimal step size.

    :param f:
    :param initial_x:
    :param full_step:
    :param expected_improve_rate:
    :param max_backtracks:
    :param accept_ratio:
    :return:
    """

    function_value = f(initial_x)

    for _, step_fraction in enumerate(0.5 ** np.arange(max_backtracks)):
        updated_x = initial_x + step_fraction * full_step
        new_function_value = f(updated_x)

        actual_improve = function_value - new_function_value
        expected_improve = expected_improve_rate * step_fraction

        improve_ratio = actual_improve / expected_improve

        if improve_ratio > accept_ratio and actual_improve > 0:
            return True, updated_x

    return False, initial_x
