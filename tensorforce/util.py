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

from tensorforce import TensorForceError


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


def cumulative_discount(values, terminals, discount, cumulative_start=0.0):
    """
    Compute cumulative discounts.
    Args:
        values: Values to discount
        terminals: Booleans indicating terminal states
        discount: Discount factor
        cumulative_start: Float or ndarray, estimated reward for state t + 1. Default 0.0

    Returns:
        dicounted_values: The cumulative discounted rewards.
    """
    if discount == 0.0:
        return np.asarray(values)

    # cumulative start can either be a number or ndarray
    if type(cumulative_start) is np.ndarray:
        discounted_values = np.zeros((len(values),) + (cumulative_start.shape))
    else:
        discounted_values = np.zeros(len(values))

    cumulative = cumulative_start
    for n, (value, terminal) in reversed(list(enumerate(zip(values, terminals)))):
        if terminal:
            cumulative = np.zeros_like(cumulative_start, dtype=np.float32)
        cumulative = value + cumulative * discount
        discounted_values[n] = cumulative

    return discounted_values


def np_dtype(dtype):
    """Translates dtype specifications in configurations to numpy data types.
    Args:
        dtype: String describing a numerical type (e.g. 'float') or numerical type primitive.

    Returns: Numpy data type

    """
    if dtype == 'float' or dtype == float or dtype == np.float32 or dtype == tf.float32:
        return np.float32
    elif dtype == 'int' or dtype == int or dtype == np.int32 or dtype == tf.int32:
        return np.int32
    elif dtype == 'bool' or dtype == bool or dtype == np.bool_ or dtype == tf.bool:
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
    if dtype == 'float' or dtype == float or dtype == np.float32 or dtype == tf.float32:
        return tf.float32
    elif dtype == 'int' or dtype == int or dtype == np.int32 or dtype == tf.int32:
        return tf.int32
    elif dtype == 'bool' or dtype == bool or dtype == np.bool_ or dtype == tf.bool:
        return tf.bool
    else:
        raise TensorForceError("Error: Type conversion from type {} not supported.".format(str(dtype)))


def get_object(obj, predefined_objects=None, default_object=None, kwargs=None):
    """
    Utility method to map some kind of object specification to its content,
    e.g. optimizer or baseline specifications to the respective classes.

    Args:
        obj: A specification dict (value for key 'type' optionally specifies
                the object, options as follows), a module path (e.g.,
                my_module.MyClass), a key in predefined_objects, or a callable
                (e.g., the class type object).
        predefined_objects: Dict containing predefined set of objects,
                accessible via their key
        default_object: Default object is no other is specified
        kwargs: Arguments for object creation

    Returns: The retrieved object

    """
    args = ()
    kwargs = dict() if kwargs is None else kwargs

    if isinstance(obj, dict):
        kwargs.update(obj)
        obj = kwargs.pop('type', None)

    if predefined_objects is not None and obj in predefined_objects:
        obj = predefined_objects[obj]
    elif isinstance(obj, str):
        module_name, function_name = obj.rsplit('.', 1)
        module = importlib.import_module(module_name)
        obj = getattr(module, function_name)
    elif callable(obj):
        pass
    elif default_object is not None:
        args = (obj,)
        obj = default_object
    else:
        # assumes the object is already instantiated
        return obj

    return obj(*args, **kwargs)
