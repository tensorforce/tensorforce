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

import tensorflow as tf

from tensorforce import TensorforceError


DTYPE_MAPPING = dict(bool=tf.bool, int=tf.int64, float=tf.float32)


def get_dtype(type):
    if type not in DTYPE_MAPPING:
        raise TensorforceError.value(
            name='tf_util.cast', argument='type', value=type,
            hint='not in {{{}}}'.format(','.join(DTYPE_MAPPING))
        )
    return DTYPE_MAPPING[type]


def dtype(x=None, dtype=None):
    for dtype, tf_dtype in DTYPE_MAPPING.items():
        if x.dtype == tf_dtype:
            return dtype
    else:
        raise TensorforceError.value(name='tf_util.dtype', argument='x.dtype', value=x.dtype)


def rank(x):
    return x.get_shape().ndims


def shape(x, unknown=-1):
    return tuple(unknown if dims is None else dims for dims in x.get_shape().as_list())


# def is_dtype(x, dtype):
#     for str_dtype, tf_dtype in tf_dtype_mapping.items():
#         if x.dtype == tf_dtype and dtype == str_dtype:
#             return True
#     else:
#         return False
#         # if x.dtype == tf.float32:
#         #     return 'float'
#         # else:
#         #     raise TensorforceError.value(name='util.dtype', argument='x', value=x.dtype)


# Conversion to generally supported TensorFlow type


def int32(x):
    if dtype(x=x) != 'int' or get_dtype(type='int') != tf.int32:
        x = tf.cast(x=x, dtype=tf.int32)
    return x


def float32(x):
    if dtype(x=x) != 'float' or get_dtype(type='float') != tf.float32:
        x = tf.cast(x=x, dtype=tf.float32)
    return x


# TensorFlow functions


def constant(value, dtype, shape=None):
    return tf.constant(value=value, dtype=get_dtype(type=dtype), shape=shape)


def zeros(shape, dtype):
    return tf.zeros(shape=shape, dtype=get_dtype(type=dtype))


def ones(shape, dtype):
    return tf.ones(shape=shape, dtype=get_dtype(type=dtype))


def identity(input):
    zero = tf.zeros_like(input=input)
    if dtype(x=zero) == 'bool':
        return tf.math.logical_or(x=input, y=zero)
    else:
        return input + zero


# def no_op():
#     return identity(input=constant(value=False, dtype='bool'))


def cast(x, dtype):
    for str_dtype, tf_dtype in DTYPE_MAPPING.items():
        if x.dtype == tf_dtype and dtype == str_dtype:
            return x
    else:
        return tf.cast(x=x, dtype=get_dtype(type=dtype))


# Other helper functions


def always_true(*args, **kwargs):
    return constant(value=True, dtype='bool')
