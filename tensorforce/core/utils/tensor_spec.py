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

from functools import total_ordering

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core.utils import tf_util


def _normalize_type(*, dtype):
    dtypes = {
        'bool': 'bool', bool: 'bool', np.bool_: 'bool', tf.bool: 'bool',
        'int': 'int', int: 'int', np.int16: 'int', np.int32: 'int', np.int64: 'int',
        tf.int16: 'int', tf.int32: 'int', tf.int64: 'int',
        'float': 'float', float: 'float', np.float16: 'float', np.float32: 'float',
        np.float64: 'float',
        tf.float16: 'float', tf.float32: 'float', tf.float64: 'float'
    }
    return dtypes.get(dtype, None)


@total_ordering
class TensorSpec(object):

    def __init__(
        self, *, type, shape=(), min_value=None, max_value=None, num_values=None, overwrite=False
    ):
        if num_values is not None and (min_value is not None or max_value is not None):
            raise TensorforceError.invalid(
                name='TensorSpec', argument='min/max_value', condition='num_values specified'
            )
        super().__setattr__('overwrite', True)
        super().__setattr__('type', None)
        if isinstance(type, tf.dtypes.DType):
            super().__setattr__('_tf_type', type)
            assert not overwrite
        self.type = type
        self.shape = shape
        if min_value is not None:
            self.min_value = min_value
        if max_value is not None:
            self.max_value = max_value
        if num_values is not None:
            self.num_values = num_values
        super().__setattr__('overwrite', overwrite)

    @property
    def rank(self):
        return len(self.shape)

    @property
    def size(self):
        return util.product(xs=self.shape)

    def py_type(self):
        if self.type == 'bool':
            return bool
        elif self.type == 'int':
            return int
        elif self.type == 'float':
            return float

    def np_type(self):
        return util.np_dtype(dtype=self.type)

    def tf_type(self):
        if hasattr(self, '_tf_type'):
            return self._tf_type
        else:
            return tf_util.DTYPE_MAPPING[self.type]

    def is_underspecified(self):
        if self.type is None or isinstance(self.type, tuple):
            return True
        elif self.shape is None or (len(self.shape) > 0 and self.shape[0] is None) or \
                any(x <= 0 for x in self.shape if x is not None):
            return True
        elif self.type == 'int' and self.num_values is not None and self.num_values <= 0:
            return True
        else:
            return False

    def json(self):
        if self.type == 'bool':
            return dict(type=self.type, shape=self.shape)

        elif self.type == 'int' and self.num_values is not None:
            return dict(type=self.type, shape=self.shape, num_values=self.num_values)

        else:
            spec = dict(type=self.type, shape=self.shape)
            if self.min_value is not None:
                if isinstance(self.min_value, np.ndarray):
                    spec['min_value'] = self.min_value.tolist()
                else:
                    spec['min_value'] = self.min_value
            if self.max_value is not None:
                if isinstance(self.max_value, np.ndarray):
                    spec['max_value'] = self.max_value.tolist()
                else:
                    spec['max_value'] = self.max_value
            return spec

    def signature(self, *, batched):
        # Check whether underspecified
        if self.is_underspecified():
            raise TensorforceError.unexpected()

        # Add leading variable-dim axis if batched
        if batched:
            shape = (None,) + self.shape
        else:
            shape = self.shape

        # TensorFlow TensorSpec
        return tf.TensorSpec(shape=tf.TensorShape(dims=shape), dtype=self.tf_type())

    def empty(self, *, batched):
        if batched:
            return tf_util.zeros(shape=((0,) + self.shape), dtype=self.type)
        else:
            return tf_util.zeros(shape=self.shape, dtype=self.type)

    def to_tensor(self, *, value, batched, recover_empty=False):
        # Check whether underspecified
        if self.is_underspecified():
            raise TensorforceError.unexpected()

        # Convert value to Numpy array, checks type
        value = np.asarray(a=value, dtype=self.np_type())

        # Check whether shape matches
        if value.shape[int(batched):] != self.shape:
            raise TensorforceError.value(
                name='TensorSpec.to_tensor', argument='value', value=value, hint='shape'
            )

        # Check for nan or inf
        if np.isnan(value).any() or np.isinf(value).any():
            raise TensorforceError.value(
                name='TensorSpec.to_tensor', argument='value', value=value, hint='is nan/inf'
            )

        # Check num_values
        if self.type == 'int' and self.num_values is not None:
            if (value < 0).any() or (value >= self.num_values).any():
                raise TensorforceError.value(
                    name='TensorSpec.to_tensor', argument='value', value=value
                )

        # Check min/max_value
        elif self.type == 'int' or self.type == 'float':
            if self.min_value is not None:
                if (value < self.min_value).any():
                    raise TensorforceError.value(
                        name='TensorSpec.to_tensor', argument='value', value=value,
                        hint='< min_value'
                    )
            if self.max_value is not None:
                if (value > self.max_value).any():
                    raise TensorforceError.value(
                        name='TensorSpec.to_tensor', argument='value', value=value,
                        hint='> max_value'
                    )

        # Convert Numpy array to TensorFlow tensor
        return tf.convert_to_tensor(value=value, dtype=self.tf_type())

    def from_tensor(self, *, tensor, batched):
        # Check whether underspecified
        if self.is_underspecified():
            raise TensorforceError.unexpected()

        # Check whether TensorFlow tensor
        if not isinstance(tensor, tf.Tensor):
            raise TensorforceError.type(
                name='TensorSpec.from_tensor', argument='tensor', dtype=type(tensor)
            )

        # Check whether tensor type and shape match
        if tf_util.dtype(x=tensor) != self.type:
            raise TensorforceError.value(
                name='TensorSpec.from_tensor', argument='tensor.dtype', value=tensor
            )
        if tf_util.shape(x=tensor)[int(batched):] != self.shape:
            raise TensorforceError.value(
                name='TensorSpec.from_tensor', argument='tensor.shape', value=tensor
            )

        # Convert tensor value to Numpy array
        value = tensor.numpy()

        # Check for nan or inf
        if np.isnan(value).any() or np.isinf(value).any():
            raise TensorforceError.value(
                name='TensorSpec.from_tensor', argument='tensor', value=value
            )

        # Check num_values
        if self.type == 'int' and self.num_values is not None:
            if (value < 0).any() or (value >= self.num_values).any():
                raise TensorforceError.value(
                    name='TensorSpec.from_tensor', argument='tensor', value=value
                )

        # Check min/max_value
        elif self.type == 'int' or self.type == 'float':
            if self.min_value is not None:
                if (value < self.min_value).any():
                    raise TensorforceError.value(
                        name='TensorSpec.from_tensor', argument='tensor', value=value
                    )
            if self.max_value is not None:
                if (value > self.max_value).any():
                    raise TensorforceError.value(
                        name='TensorSpec.from_tensor', argument='tensor', value=value
                    )

        # If singleton shape, return Python object instead of Numpy array
        if self.shape == () and not batched:
            value = value.item()

        return value

    def tf_assert(self, *, x, batch_size=None, include_type_shape=False, message=None):
        if not isinstance(x, (tf.Tensor, tf.Variable)):
            raise TensorforceError.type(name='TensorSpec.tf_assert', argument='x', dtype=type(x))

        if batch_size is None:
            pass
        elif not isinstance(batch_size, tf.Tensor):
            raise TensorforceError.type(
                name='TensorSpec.tf_assert', argument='batch_size', dtype=type(batch_size)
            )
        elif tf_util.dtype(x=batch_size) != 'int' or tf_util.shape(x=batch_size) != ():
            raise TensorforceError.value(
                name='TensorSpec.tf_assert', argument='batch_size', value=batch_size
            )
        assertions = list()

        if message is not None and '{name}' in message:
            message = message.format(name='', issue='{issue}')

        # Type
        tf.debugging.assert_type(
            tensor=x, tf_type=self.tf_type(),
            message=(None if message is None else message.format(issue='type'))
        )

        # Shape
        shape = tf_util.constant(value=self.shape, dtype='int')
        if batch_size is not None:
            shape = tf.concat(values=(tf.expand_dims(input=batch_size, axis=0), shape), axis=0)
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf_util.cast(x=tf.shape(input=x), dtype='int'), y=shape,
                    message=(None if message is None else message.format(issue='shape'))
                )
            )

        if self.type == 'float':
            assertions.append(tf.debugging.assert_all_finite(
                x=x, message=('' if message is None else message.format(issue='inf/nan value'))
            ))

        # Min/max value (includes num_values)
        if self.type != 'bool' and self.min_value is not None:
            assertions.append(tf.debugging.assert_greater_equal(
                x=x, y=tf_util.constant(value=self.min_value, dtype=self.type),
                message=(None if message is None else message.format(issue='min value'))
            ))
        if self.type != 'bool' and self.max_value is not None:
            assertions.append(tf.debugging.assert_less_equal(
                x=x, y=tf_util.constant(value=self.max_value, dtype=self.type),
                message=(None if message is None else message.format(issue='max value'))
            ))

        return assertions

    def unify(self, *, other, name='TensorSpec.unify'):
        # Unify type
        if self.type is None:
            dtype = other.type
        elif other.type is None:
            dtype = self.type
        elif util.is_iterable(x=self.type):
            if util.is_iterable(x=other.type):
                if set(self.type) <= set(other.type):
                    dtype = self.type
                elif set(other.type) <= set(self.type):
                    dtype = other.type
                else:
                    raise TensorforceError.mismatch(
                        name=name, argument='type', value1=self.type, value2=other.type
                    )
            elif other.type in self.type:
                dtype = other.type
            else:
                raise TensorforceError.mismatch(
                    name=name, argument='type', value1=self.type, value2=other.type
                )
        elif util.is_iterable(x=other.type):
            if self.type in other.type:
                dtype = self.type
            else:
                raise TensorforceError.mismatch(
                    name=name, argument='type', value1=self.type, value2=other.type
                )
        elif self.type == other.type:
            dtype = self.type
        else:
            raise TensorforceError.mismatch(
                name=name, argument='type', value1=self.type, value2=other.type
            )

        # Unify shape
        if self.shape is None:
            shape = other.shape
        elif other.shape is None:
            shape = self.shape
        else:
            reverse_shape = list()
            start = len(self.shape) - 1
            if self.shape[-1] is None:
                reverse_shape.extend(other.shape[len(self.shape) - 1:])
                start = len(self.shape) - 2
            elif other.shape[-1] is None:
                reverse_shape.extend(self.shape[len(other.shape) - 1:])
                start = len(other.shape) - 2
            elif len(self.shape) != len(other.shape):
                raise TensorforceError.mismatch(
                    name=name, argument='rank', value1=self.rank, value2=other.rank
                )
            for n in range(start, -1, -1):
                if self.shape[n] == 0:
                    reverse_shape.append(other.shape[n])
                elif other.shape[n] == 0:
                    reverse_shape.append(self.shape[n])
                elif self.shape[n] == -1 and other.shape[n] > 0:
                    reverse_shape.append(other.shape[n])
                elif other.shape[n] == -1 and self.shape[n] > 0:
                    reverse_shape.append(self.shape[n])
                elif self.shape[n] == other.shape[n]:
                    reverse_shape.append(self.shape[n])
                else:
                    raise TensorforceError.mismatch(
                        name=name, argument='shape', value1=self.shape, value2=other.shape
                    )
            shape = tuple(reversed(reverse_shape))

        # Unify min_value
        if dtype == 'bool':
            min_value = None
        elif self.type != 'bool' and self.min_value is not None:
            if other.type != 'bool' and other.min_value is not None:
                if isinstance(self.min_value, np.ndarray) or \
                        isinstance(other.min_value, np.ndarray):
                    min_value = np.minimum(self.min_value, other.min_value)
                elif self.min_value < other.min_value:
                    min_value = other.min_value
                else:
                    min_value = self.min_value
            else:
                min_value = self.min_value
        elif other.type != 'bool' and other.min_value is not None:
            min_value = other.min_value
        else:
            min_value = None

        # Unify max_value
        if dtype == 'bool':
            max_value = None
        elif self.type != 'bool' and self.max_value is not None:
            if other.type != 'bool' and other.max_value is not None:
                if isinstance(self.max_value, np.ndarray) or \
                        isinstance(other.max_value, np.ndarray):
                    max_value = np.maximum(self.max_value, other.max_value)
                elif self.max_value < other.max_value:
                    max_value = other.max_value
                else:
                    max_value = self.max_value
            else:
                max_value = self.max_value
        elif other.type != 'bool' and other.max_value is not None:
            max_value = other.max_value
        else:
            max_value = None
        if min_value is not None and max_value is not None:
            if isinstance(min_value, np.ndarray) or isinstance(max_value, np.ndarray):
                if (min_value > max_value).any():
                    raise TensorforceError.mismatch(
                        name=name, argument='min/max_value', value1=min_value, value2=max_value
                    )
            else:
                if min_value > max_value:
                    raise TensorforceError.mismatch(
                        name=name, argument='min/max_value', value1=min_value, value2=max_value
                    )

        # Unify num_values
        if dtype != 'int' and (not isinstance(dtype, tuple) or 'int' not in dtype):
            num_values = None
        elif self.type == 'int' and self.num_values is not None:
            if other.type == 'int' and other.num_values is not None:
                if self.num_values == 0:
                    num_values = other.num_values
                elif other.num_values == 0:
                    num_values = self.num_values
                elif self.num_values == other.num_values:
                    num_values = self.num_values
                else:
                    raise TensorforceError.mismatch(
                        name=name, argument='num_values', value1=self.num_values,
                        value2=other.num_values
                    )
            else:
                num_values = self.num_values
        elif other.type == 'int' and other.num_values is not None:
            num_values = other.num_values
        else:
            num_values = None
        if num_values is not None:
            min_value = None
            max_value = None

        # Unified tensor spec
        return TensorSpec(
            type=dtype, shape=shape, min_value=min_value, max_value=max_value, num_values=num_values
        )

    # def __len__(self):
    #     return 1

    # def __iter__(self):
    #     return
    #     yield

    # def values(self):
    #     yield self

    # def items(self):
    #     yield None, self

    # def value(self):
    #     return self

    def copy(self, *, overwrite=None):
        if overwrite is None:
            overwrite = self.overwrite

        if self.type == 'bool':
            return TensorSpec(type=self.type, shape=self.shape, overwrite=overwrite)

        elif self.type == 'int' and self.num_values is not None:
            return TensorSpec(
                type=self.type, shape=self.shape, num_values=self.num_values, overwrite=overwrite
            )

        else:
            return TensorSpec(
                type=self.type, shape=self.shape, min_value=self.min_value,
                max_value=self.max_value, overwrite=overwrite
            )

    # def fmap(self, *, function, cls=None, with_names=False, zip_values=None):
    #     args = (self,)

    #     # with_names
    #     if with_names:
    #         args = (None,) + args

    #     # zip_values
    #     if isinstance(zip_values, (tuple, list)):
    #         for value in zip_values:
    #             if isinstance(value, NestedDict):
    #                 assert len(value) == 1 and None in value
    #                 args += (value[None],)
    #             else:
    #                 args += (value,)
    #     elif isinstance(zip_values, NestedDict):
    #         assert len(zip_values) == 1 and None in zip_values
    #         args += (zip_values[None],)
    #     elif zip_values is not None:
    #         args += (zip_values,)

    #     # Actual value mapping
    #     value = function(*args)

    #     # from tensorforce.core import TensorDict
    #     if cls is None:  # or issubclass(cls, TensorDict):
    #         # Use same class
    #         return value

    #     elif issubclass(cls, list):
    #         # Special target class list implies flatten
    #         values = cls()
    #         if isinstance(value, cls):
    #             values.extend(value)
    #         else:
    #             values.append(value)
    #         return values

    #     elif issubclass(cls, dict):
    #         # Custom target class
    #         values = cls()
    #         values[None] = value
    #         return values

    #     else:
    #         raise TensorforceError.value(name='TensorSpec.fmap', argument='cls', value=cls)

    def __setattr__(self, name, value):
        if not self.overwrite:
            raise NotImplementedError

        if name == 'type':
            if value is None:
                # Type: None
                pass
            elif util.is_iterable(x=value):
                # Type: tuple(*types)
                if any(_normalize_type(dtype=x) is None for x in value):
                    raise TensorforceError.value(name='TensorSpec', argument=name, value=value)
                value = tuple(_normalize_type(dtype=x) for x in value)
            else:
                # Type: 'bool' | 'int' | 'float'
                if _normalize_type(dtype=value) is None:
                    raise TensorforceError.value(name='TensorSpec', argument=name, value=value)
                value = _normalize_type(dtype=value)

            # Delete attributes not required anymore
            if self.type is not None and self.type != 'bool' and value == 'bool':
                super().__delattr__('min_value')
                super().__delattr__('max_value')
            if self.type is not None and (
                self.type == 'int' or (isinstance(self.type, tuple) and 'int' in self.type)
            ) and value != 'int' and (not isinstance(value, tuple) or 'int' not in value):
                super().__delattr__('num_values')

            # Set type attribute
            super().__setattr__(name, value)

            # Reset attributes
            if self.type == 'int' or (isinstance(self.type, tuple) and 'int' in self.type):
                self.min_value = None
                self.max_value = None
                self.num_values = None
            elif self.type != 'bool':
                self.min_value = None
                self.max_value = None

        elif name == 'shape':
            if value is None:
                # Shape: None
                pass
            elif util.is_iterable(x=value):
                if len(value) > 0 and value[0] is None:
                    # Shape: tuple(None, *ints >= -1)
                    try:
                        value = (None,) + tuple(int(x) for x in value[1:])
                        if any(x < -1 for x in value[1:]):
                            raise TensorforceError.value(
                                name='TensorSpec', argument=name, value=value
                            )
                    except BaseException:
                        raise TensorforceError.type(
                            name='TensorSpec', argument=name, value=type(value)
                        )
                else:
                    # Shape: tuple(*ints >= -1)
                    try:
                        value = tuple(int(x) for x in value)
                        if any(x < -1 for x in value):
                            raise TensorforceError.value(
                                name='TensorSpec', argument=name, value=value
                            )
                    except BaseException:
                        raise TensorforceError.value(name='TensorSpec', argument=name, value=value)
            else:
                # Shape: (int >= -1,)
                try:
                    value = (int(value),)
                    if value[0] < -1:
                        raise TensorforceError.value(name='TensorSpec', argument=name, value=value)
                except BaseException:
                    raise TensorforceError.type(name='TensorSpec', argument=name, value=type(value))

            # TODO: check min/max_value shape if np.ndarray

            # Set shape attribute
            super().__setattr__(name, value)

        elif name == 'min_value' or name == 'max_value':
            # Invalid for type == 'bool', or type == 'int' and num_values != None
            if self.type == 'bool':
                raise TensorforceError.invalid(
                    name='TensorSpec', argument=name, condition='type is bool'
                )

            if value is None:
                # Min/max value: None
                pass
            else:
                # Min/max value: int/float
                try:
                    value = self.py_type()(value)
                    if self.type == 'int' and self.num_values is not None:
                        if name == 'min_value':
                            assert value == 0
                        elif name == 'max_value':
                            assert value == self.num_values - 1
                except BaseException:
                    try:
                        value = np.asarray(value, dtype=self.np_type())
                        if self.type == 'int':
                            assert self.num_values is None
                    except BaseException:
                        raise TensorforceError.type(
                            name='TensorSpec', argument=name, value=type(value)
                        )

                if isinstance(value, np.ndarray):
                    if self.shape is not None and (
                        value.ndim > len(self.shape) or value.shape != self.shape[:value.ndim]
                    ):
                        raise TensorforceError.value(
                            name='TensorSpec', argument=(name + ' shape'), value=value.shape,
                            hint='incompatible with {}'.format(self.shape)
                        )
                    if name == 'min_value' and self.max_value is not None and \
                            (value > self.max_value - util.epsilon).any():
                        raise TensorforceError.value(
                            name='TensorSpec', argument=name, value=value,
                            condition='max_value = {}'.format(self.max_value)
                        )
                    elif name == 'max_value' and self.min_value is not None and \
                            (value < self.min_value + util.epsilon).any():
                        raise TensorforceError.value(
                            name='TensorSpec', argument=name, value=value,
                            condition='min_value = {}'.format(self.min_value)
                        )
                else:
                    if name == 'min_value' and self.max_value is not None:
                        if isinstance(self.max_value, np.ndarray):
                            if (value > self.max_value - util.epsilon).any():
                                raise TensorforceError.value(
                                    name='TensorSpec', argument=name, value=value,
                                    condition='max_value = {}'.format(self.max_value)
                                )
                        elif value > self.max_value - util.epsilon:
                            raise TensorforceError.value(
                                name='TensorSpec', argument=name, value=value,
                                condition='max_value = {}'.format(self.max_value)
                            )
                    elif name == 'max_value' and self.min_value is not None:
                        if isinstance(self.min_value, np.ndarray):
                            if (value < self.min_value + util.epsilon).any():
                                raise TensorforceError.value(
                                    name='TensorSpec', argument=name, value=value,
                                    condition='min_value = {}'.format(self.min_value)
                                )
                        elif value < self.min_value + util.epsilon:
                            raise TensorforceError.value(
                                name='TensorSpec', argument=name, value=value,
                                condition='min_value = {}'.format(self.min_value)
                            )

            # Set min/max_value attribute
            super().__setattr__(name, value)

        elif name == 'num_values':
            # Invalid for type != 'int'
            if self.type != 'int' and (not isinstance(self.type, tuple) or 'int' not in self.type):
                raise TensorforceError.invalid(
                    name='TensorSpec', argument=name, condition='type is {}'.format(self.type)
                )

            if value is None:
                # Num values: None
                pass
            else:
                # Num values: int >= 0
                try:
                    value = int(value)
                except BaseException:
                    raise TensorforceError.type(name='TensorSpec', argument=name, value=type(value))
                if value < 0:
                    raise TensorforceError.value(name='TensorSpec', argument=name, value=value)

            # Set num_values attribute and min/max_value accordingly
            super().__setattr__(name, value)
            if value is not None and value > 0:
                super().__setattr__('min_value', 0)
                super().__setattr__('max_value', value - 1)
            else:
                super().__setattr__('min_value', None)
                super().__setattr__('max_value', None)

        else:
            raise TensorforceError.invalid(name='TensorSpec', argument=name)

    def __repr__(self):
        if self.type == 'int' and self.num_values is not None:
            return 'TensorSpec(type={type}, shape={shape}, num_values={num_values})'.format(
                type=self.type, shape=self.shape, num_values=self.num_values
            )
        elif self.type != 'bool' and self.min_value is not None:
            if self.max_value is None:
                return 'TensorSpec(type={type}, shape={shape}, min_value={min_value})'.format(
                    type=self.type, shape=self.shape, min_value=self.min_value
                )
            else:
                return ('TensorSpec(type={type}, shape={shape}, min_value={min_value}, max_value='
                        '{max_value})').format(
                    type=self.type, shape=self.shape, min_value=self.min_value,
                    max_value=self.max_value
                )
        elif self.type != 'bool' and self.max_value is not None:
            return 'TensorSpec(type={type}, shape={shape}, max_value={max_value})'.format(
                type=self.type, shape=self.shape, max_value=self.max_value
            )
        else:
            return 'TensorSpec(type={type}, shape={shape})'.format(type=self.type, shape=self.shape)

    __str__ = __repr__

    def tuple(self):
        return (
            self.type, self.shape, getattr(self, 'min_value', None),
            getattr(self, 'max_value', None), getattr(self, 'num_values', None)
        )

    def __hash__(self):
        return hash(self.tuple())

    def __eq__(self, other):
        return isinstance(other, TensorSpec) and self.tuple() == other.tuple()

    def __lt__(self, other):
        if not isinstance(other, TensorSpec):
            return NotImplementedError
        return self.tuple() < other.tuple()

    def __delattr__(self, name):
        raise NotImplementedError
