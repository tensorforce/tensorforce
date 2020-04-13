# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from collections import OrderedDict
from datetime import datetime
import logging

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError


epsilon = 1e-6


log_levels = dict(
    info=logging.INFO,
    debug=logging.DEBUG,
    critical=logging.CRITICAL,
    warning=logging.WARNING,
    fatal=logging.FATAL
)


def debug(message):
    logging.warning('{}: {}'.format(datetime.now().strftime('%H:%M:%S-%f')[:-3], message))


def is_iterable(x):
    if isinstance(x, (str, dict, np.ndarray, tf.Tensor)):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


def unary_tuple(x, depth):
    assert depth > 0
    for _ in range(depth):
        x = (x,)
    return x


def product(xs, empty=1):
    """Computes the product along the elements in an iterable.

    Args:
        xs: Iterable containing numbers.
        empty: ??

    Returns: Product along iterable.

    """
    result = None
    for x in xs:
        if result is None:
            result = x
        else:
            result *= x

    if result is None:
        result = empty

    return result


def fmap(function, xs, depth=-1, map_keys=False, map_types=()):
    if xs is None and None not in map_types:
        assert depth <= 0
        return None
    elif isinstance(xs, tuple) and depth != 0 and tuple not in map_types:
        return tuple(
            fmap(function=function, xs=x, depth=(depth - 1), map_keys=map_keys, map_types=map_types)
            for x in xs
        )
    elif isinstance(xs, list) and depth != 0 and list not in map_types:
        return [
            fmap(function=function, xs=x, depth=(depth - 1), map_keys=map_keys, map_types=map_types)
            for x in xs
        ]
    elif isinstance(xs, set) and depth != 0 and set not in map_types:
        return {
            fmap(function=function, xs=x, depth=(depth - 1), map_keys=map_keys, map_types=map_types)
            for x in xs
        }
    elif isinstance(xs, OrderedDict) and depth != 0 and OrderedDict not in map_types:
        if map_keys:
            return OrderedDict((
                (function(key), fmap(
                    function=function, xs=x, depth=(depth - 1), map_keys=map_keys,
                    map_types=map_types
                )) for key, x in xs.items()
            ))
        else:
            return OrderedDict((
                (key, fmap(
                    function=function, xs=x, depth=(depth - 1), map_keys=map_keys,
                    map_types=map_types
                )) for key, x in xs.items()
            ))
    elif isinstance(xs, dict) and depth != 0 and dict not in map_types:
        if map_keys:
            return {
                function(key): fmap(
                    function=function, xs=x, depth=(depth - 1), map_keys=map_keys,
                    map_types=map_types
                ) for key, x in xs.items()
            }
        else:
            return {
                key: fmap(
                    function=function, xs=x, depth=(depth - 1), map_keys=map_keys,
                    map_types=map_types
                ) for key, x in xs.items()
            }
    else:
        if map_keys:  # or depth <= 0?
            return xs
        else:
            assert depth <= 0
            return function(xs)


def reduce_all(predicate, xs):
    if xs is None:
        return False
    elif isinstance(xs, tuple):
        return all(reduce_all(predicate=predicate, xs=x) for x in xs)
    elif isinstance(xs, list):
        return all(reduce_all(predicate=predicate, xs=x) for x in xs)
    elif isinstance(xs, set):
        return all(reduce_all(predicate=predicate, xs=x) for x in xs)
    elif isinstance(xs, dict):
        return all(reduce_all(predicate=predicate, xs=x) for x in xs.values())  
    else:
        return predicate(xs)


def flatten(xs):
    if xs is None:
        return None
    elif isinstance(xs, (tuple, list, set)):
        return [x for ys in xs for x in flatten(xs=ys)]
    elif isinstance(xs, dict):
        return [x for ys in xs.values() for x in flatten(xs=ys)]
    else:
        return [xs]


def zip_items(*args):
    # assert len(args) > 0 and all(arg is None or isinstance(arg, dict) for arg in args)
    # assert args[0] is not None
    # for key in args[0]:
    #     key_values = (key,) + tuple(None if arg is None else arg[key] for arg in args)
    #     yield key_values
    assert len(args) > 0
    assert all(isinstance(arg, dict) and len(arg) == len(args[0]) for arg in args)
    for key in args[0]:
        key_values = (key,) + tuple(arg[key] for arg in args)
        yield key_values


def deep_equal(xs, ys):
    if isinstance(xs, dict):
        if not isinstance(ys, dict):
            return False
        for _, x, y in zip_items(xs, ys):
            if not deep_equal(xs=x, ys=y):
                return False
        return True
    elif is_iterable(x=xs):
        if not is_iterable(x=ys):
            return False
        for x, y in zip(xs, ys):
            if not deep_equal(xs=x, ys=y):
                return False
        return True
    else:
        return xs == ys


def deep_disjoint_update(target, source):  # , ignore=()
    for key, value in source.items():
        if key not in target:
            target[key] = value
        # elif key in ignore:
        #     continue
        elif isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise TensorforceError.mismatch(
                    name='spec', argument=key, value1=target[key], value2=value
                )
            deep_disjoint_update(target=target[key], source=value)
        elif is_iterable(x=target[key]):
            if not is_iterable(x=value) or len(target[key]) != len(value):
                raise TensorforceError.mismatch(
                    name='spec', argument=key, value1=target[key], value2=value
                )
            for x, y in zip(target[key], value):
                if x != y:
                    raise TensorforceError.mismatch(
                        name='spec', argument=key, value1=target[key], value2=value
                    )
        elif target[key] != value:
            raise TensorforceError.mismatch(
                name='spec', argument=key, value1=target[key], value2=value
            )


def py_dtype(dtype):
    if dtype == 'float':  # or dtype == float or dtype == np.float32 or dtype == tf.float32:
        return float
    elif dtype == 'int' or dtype == 'long':
    # dtype == int or dtype == np.int32 or dtype == tf.int32 or
    # or dtype == np.int64 or dtype == tf.int64
        return int
    elif dtype == 'bool':  # or dtype == bool or dtype == np.bool_ or dtype == tf.bool:
        return bool
    else:
        raise TensorforceError.value(name='util.py_dtype', argument='dtype', value=dtype)


np_dtype_mapping = dict(bool=np.bool_, int=np.int64, long=np.int64, float=np.float32)


def np_dtype(dtype):
    """Translates dtype specifications in configurations to numpy data types.
    Args:
        dtype: String describing a numerical type (e.g. 'float') or numerical type primitive.

    Returns: Numpy data type

    """
    if dtype in np_dtype_mapping:
        return np_dtype_mapping[dtype]
    else:
        raise TensorforceError.value(name='util.np_dtype', argument='dtype', value=dtype)



reverse_dtype_mapping = {
    bool: 'bool', np.bool_: 'bool', tf.bool: 'bool',
    int: 'int', np.int32: 'int', tf.int32: 'int',
    np.int64: 'int', tf.int64: 'int',
    float: 'float', np.float32: 'float', tf.float32: 'float'
}


# def tf_dtype(dtype):
#     """Translates dtype specifications in configurations to tensorflow data types.

#        Args:
#            dtype: String describing a numerical type (e.g. 'float'), numpy data type,
#                or numerical type primitive.

#        Returns: TensorFlow data type

#     """
#     if dtype in tf_dtype_mapping:
#         return tf_dtype_mapping[dtype]
#     else:
#         raise TensorforceError.value(name='util.tf_dtype', argument='dtype', value=dtype)


def get_tensor_dependencies(tensor):
    """
    Utility method to get all dependencies (including placeholders) of a tensor (backwards through the graph).

    Args:
        tensor (tf.Tensor): The input tensor.

    Returns: Set of all dependencies (including needed placeholders) for the input tensor.
    """
    dependencies = set()
    dependencies.update(tensor.op.inputs)
    for sub_op in tensor.op.inputs:
        dependencies.update(get_tensor_dependencies(sub_op))
    return dependencies


reserved_names = {
    'states', 'actions', 'terminal', 'reward', 'deterministic', 'optimization',
    # Types
    'bool', 'int', 'long', 'float',
    # Value specification attributes
    'shape', 'type', 'num_values', 'min_value', 'max_value'
    # Special values?
    'equal', 'loss', 'same', 'x'
}


def join_scopes(*args):
    return '/'.join(args)


def is_valid_name(name):
    if not isinstance(name, str):
        return False
    if name == '':
        return False
    if '/' in name:
        return False
    if '.' in name:
        return False
    if name in reserved_names:
        return False
    return True


def is_nested(name):
    return name in ('states', 'internals', 'auxiliaries', 'actions')
