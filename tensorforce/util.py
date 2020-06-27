# Copyright 2020 Tensorforce Team. All Rights Reserved.
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


class NullContext(object):

    singleton = None

    def __new__(cls):
        if cls.singleton is None:
            cls.singleton = super().__new__(cls)
        return cls.singleton

    def __enter__(self):
        return self

    def __exit__(self, etype, exception, traceback):
        pass

    def __getattr__(self, name):
        raise NotImplementedError

    def __setattr__(self, name, value):
        raise NotImplementedError

    def __delattr__(self, name):
        raise NotImplementedError


def debug(message):
    logging.warning('{}: {}'.format(datetime.now().strftime('%H:%M:%S-%f')[:-3], message))


def overwrite_staticmethod(obj, function):
    qualname = getattr(obj, function).__qualname__

    def overwritten(*args, **kwargs):
        raise TensorforceError(message="Function {}() is a static method.".format(qualname))

    setattr(obj, function, overwritten)


def is_iterable(x):
    if isinstance(x, (str, dict, np.ndarray, tf.Tensor)):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


def is_equal(x, y):
    if isinstance(x, tuple):
        return isinstance(y, tuple) and all(is_equal(x=x, y=y) for x, y in zip(x, y))
    elif isinstance(x, (list, tuple)):
        return isinstance(y, list) and all(is_equal(x=x, y=y) for x, y in zip(x, y))
    elif isinstance(x, dict):
        return isinstance(y, dict) and len(x) == len(y) and \
            all(k in y and is_equal(x=v, y=y[k]) for k, v in x.items())
    elif isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and (x == y).all()
    else:
        return x == y


def unary_tuple(x, depth):
    assert depth > 0
    for _ in range(depth):
        x = (x,)
    return x


def product(xs, empty=1):
    result = None
    for x in xs:
        if result is None:
            result = x
        else:
            result *= x

    if result is None:
        result = empty

    return result


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
