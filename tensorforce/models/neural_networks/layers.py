# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Creates various neural network layers. For most layers, these functions use
TF-slim layer types. The purpose of this class is to encapsulate
layer types to mix between layers available in TF-slim and custom implementations.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy

import tensorflow as tf
from tensorforce.util.config_util import make_function
import numpy as np

tf_slim = tf.contrib.slim


def layer_wrapper(layer_constructor, reshape=None):
    def layer_fn(input, config, scope=None):
        """
        Initialize layer of any kind.
        :param input: input layer
        :param config: layer configuration
        :param scope: tensorflow scope
        """
        kwargs = copy.copy(config)

        # Remove `type` from kwargs array.
        kwargs.pop("type", None)

        for fk in ["weights_initializer", "weights_regularizer", "biases_initializer", "activation_fn", "normalizer_fn"]:
            make_function(kwargs, fk)

        # Force our own scope definitions
        kwargs['scope'] = scope

        if reshape and callable(reshape):
            input = reshape(input)

        return layer_constructor(input, **kwargs)

    return layer_fn

dense = layer_wrapper(tf_slim.fully_connected)
conv2d = layer_wrapper(tf_slim.conv2d)
linear = layer_wrapper(tf_slim.linear, reshape=lambda input: tf.reshape(input, (-1, int(np.prod(input.get_shape()[1:])))))

layers = {
    'dense': dense,
    'conv2d': conv2d,
    'linear': linear
}
