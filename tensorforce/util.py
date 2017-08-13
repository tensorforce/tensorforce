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

import importlib
import logging
import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, Configuration


epsilon = 1e-6


log_levels = dict(
    info=logging.INFO,
    debug=logging.DEBUG,
    critical=logging.CRITICAL,
    warning=logging.WARNING,
    fatal=logging.FATAL
)


def prod(xs):
    """Computes the product along the elements in an iterable. Returns 1 for empty
        iterable.
    Args:
        xs: Iterable containing numbers.

    Returns: Product along iterable.

    """
    p = 1
    for x in xs:
        p *= x
    return p


def rank(x):
    return x.get_shape().ndims


def shape(x, unknown=-1):
    return tuple(unknown if dims is None else dims for dims in x.get_shape().as_list())


def cumulative_discount(rewards, terminals, discount, cumulative_start=0.0):
    """
    Compute cumulative discounts.
    Args:
        rewards: Rewards from a rollout
        terminals: Booleans indicating terminal states
        discount: Discount factor

    Returns:

    """
    if discount == 0.0:
        return np.asarray(rewards)
    cumulative = cumulative_start
    discounted_rewards = np.zeros(len(rewards))

    for n, (reward, terminal) in reversed(list(enumerate(zip(rewards, terminals)))):
        if terminal:
            cumulative = 0.0
        cumulative = reward + cumulative * discount
        discounted_rewards[n] = cumulative

    return discounted_rewards


def np_dtype(dtype):
    """Translates dtype specifications in configurations to numpy data types.
    Args:
        dtype: String describing a numerical type (e.g. 'float') or numerical type primitive.

    Returns: Numpy data type

    """
    if dtype == 'float' or dtype == float:
        return np.float32
    elif dtype == 'int' or dtype == int:
        return np.int32
    elif dtype == 'bool' or dtype == bool:
        return np.bool_
    else:
        raise TensorForceError("Error: Type conversion from type {} not supported.".format(str(dtype)))


def tf_dtype(dtype):
    """Translates dtype specifications in configurations to tensorflow data types.
       Args:
           dtype: String describing a numerical type (e.g. 'float'), numpy data type,
            or numerical type primitive.

       Returns: TensorFlow data type

       """
    if dtype == 'float' or dtype == float or dtype == np.float32:
        return tf.float32
    elif dtype == 'int' or dtype == int or dtype == np.int32:
        return tf.int32
    else:
        raise TensorForceError("Error: Type conversion from type {} not supported.".format(str(dtype)))


def get_function(fct, predefined=None):
    """
    Turn a function specification from a configuration to a function, e.g.
    by importing it from the respective module.
    Args:
        fct:
        predefined:

    Returns:

    """
    if predefined is not None and fct in predefined:
        return predefined[fct]
    elif isinstance(fct, str):
        module_name, function_name = fct.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    elif callable(fct):
        return fct
    else:
        raise TensorForceError("Argument {} cannot be turned into a function.".format(fct))


def get_object(obj, predefined=None, kwargs=None):
    """
    Utility method to map Configuration objects to their contents, e.g. optimisers, baselines
    to the respective classes.

    Args:
        obj: Configuration object or dict
        predefined: Default value
        kwargs: Parameters for the configuration

    Returns: The retrieved object

    """
    if isinstance(obj, Configuration):
        fct = obj.type
        full_kwargs = {key: value for key, value in obj if key != 'type'}
    elif isinstance(obj, dict):
        fct = obj['type']
        full_kwargs = {key: value for key, value in obj.items() if key != 'type'}
    else:
        fct = obj
        full_kwargs = dict()
    obj = get_function(fct=fct, predefined=predefined)
    if kwargs is not None:
        full_kwargs.update(kwargs)
    return obj(**full_kwargs)
