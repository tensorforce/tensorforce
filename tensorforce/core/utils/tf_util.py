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

import tensorflow as tf

from tensorforce import TensorforceError


DTYPE_MAPPING = dict(bool=tf.dtypes.bool, int=tf.dtypes.int64, float=tf.dtypes.float32)


def is_tensor(*, x):
    return isinstance(x, (tf.IndexedSlices, tf.Tensor, tf.Variable))


def get_dtype(*, type):
    if type not in DTYPE_MAPPING:
        raise TensorforceError.value(
            name='tf_util.cast', argument='type', value=type,
            hint='not in {{{}}}'.format(','.join(DTYPE_MAPPING))
        )
    return DTYPE_MAPPING[type]


def dtype(*, x=None, dtype=None, fallback_tf_dtype=False):
    for dtype, tf_dtype in DTYPE_MAPPING.items():
        if x.dtype == tf_dtype:
            return dtype
    else:
        if fallback_tf_dtype:
            return x.dtype
        raise TensorforceError.value(name='tf_util.dtype', argument='x.dtype', value=x.dtype)


def rank(*, x):
    return x.get_shape().ndims


def shape(*, x, unknown=-1):
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


def constant(*, value, dtype, shape=None):
    return tf.constant(value=value, dtype=get_dtype(type=dtype), shape=shape)


def zeros(*, shape, dtype):
    return tf.zeros(shape=shape, dtype=get_dtype(type=dtype))


def ones(*, shape, dtype):
    return tf.ones(shape=shape, dtype=get_dtype(type=dtype))


def identity(input, name=None):
    zero = tf.zeros_like(input=input)
    if zero.dtype is tf.bool:
        return tf.math.logical_or(x=input, y=zero, name=name)
    else:
        return tf.math.add(x=input, y=zero, name=name)


# def no_op():
#     return identity(input=constant(value=False, dtype='bool'))


def cast(*, x, dtype):
    for str_dtype, tf_dtype in DTYPE_MAPPING.items():
        if x.dtype == tf_dtype and dtype == str_dtype:
            return x
    else:
        return tf.cast(x=x, dtype=get_dtype(type=dtype))


# Other helper functions


def always_true(*args, **kwargs):
    return constant(value=True, dtype='bool')


def lift_indexedslices(binary_op, x, y, with_assertions):
    if isinstance(x, tf.IndexedSlices):
        assert isinstance(y, tf.IndexedSlices)
        assertions = list()
        if with_assertions:
            assertions.append(tf.debugging.assert_equal(x=x.indices, y=y.indices))
        with tf.control_dependencies(control_inputs=assertions):
            return tf.IndexedSlices(
                values=binary_op(x.values, y.values), indices=x.indices, dense_shape=x.dense_shape
            )
    else:
        return binary_op(x, y)
