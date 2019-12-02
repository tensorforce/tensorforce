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
    if isinstance(x, (str, dict, tf.Tensor)):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


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


def compose(function1, function2):
    def composed(*args, **kwargs):
        return function1(function2(*args, **kwargs))
    return composed


def tf_always_true(*args, **kwargs):
    return tf.constant(value=True, dtype=tf_dtype(dtype='bool'))


def fmap(function, xs, depth=-1):
    if xs is None:
        assert depth <= 0
        return None
    elif isinstance(xs, tuple) and depth != 0:
        return tuple(fmap(function=function, xs=x, depth=(depth - 1)) for x in xs)
    elif isinstance(xs, list) and depth != 0:
        return [fmap(function=function, xs=x, depth=(depth - 1)) for x in xs]
    elif isinstance(xs, set) and depth != 0:
        return {fmap(function=function, xs=x, depth=(depth - 1)) for x in xs}
    elif isinstance(xs, OrderedDict) and depth != 0:
        return OrderedDict(
            ((key, fmap(function=function, xs=x, depth=(depth - 1))) for key, x in xs.items())
        )
    elif isinstance(xs, dict) and depth != 0:
        return {key: fmap(function=function, xs=x, depth=(depth - 1)) for key, x in xs.items()}
    else:
        assert depth <= 0
        return function(xs)


def not_nan_inf(x):
    return not np.isnan(x).any() and not np.isinf(x).any()


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


def dtype(x):
    for dtype, tf_dtype in tf_dtype_mapping.items():
        if x.dtype == tf_dtype:
            return dtype
    else:
        if x.dtype == tf.float32:
            return 'float'
        else:
            raise TensorforceError.value(name='dtype', value=x.dtype)


def rank(x):
    return x.get_shape().ndims


def shape(x, unknown=-1):
    return tuple(unknown if dims is None else dims for dims in x.get_shape().as_list())


def no_operation():
    # Operation required, constant not enough.
    # Returns false
    return identity_operation(x=tf.constant(value=False, dtype=tf_dtype(dtype='bool')))


def identity_operation(x, operation_name=None):
    zero = tf.zeros_like(input=x)
    if dtype(x=zero) == 'bool':
        x = tf.math.logical_or(x=x, y=zero, name=operation_name)
    elif dtype(x=zero) in ('int', 'long', 'float'):
        x = tf.math.add(x=x, y=zero, name=operation_name)
    else:
        raise TensorforceError.value(name='tensor', value=x)
    return x


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
        raise TensorforceError.value(name='dtype', value=dtype)


np_dtype_mapping = dict(bool=np.bool_, int=np.int32, long=np.int64, float=np.float32)


def np_dtype(dtype):
    """Translates dtype specifications in configurations to numpy data types.
    Args:
        dtype: String describing a numerical type (e.g. 'float') or numerical type primitive.

    Returns: Numpy data type

    """
    if dtype in np_dtype_mapping:
        return np_dtype_mapping[dtype]
    else:
        raise TensorforceError.value(name='dtype', value=dtype)


tf_dtype_mapping = dict(bool=tf.bool, int=tf.int32, long=tf.int64, float=tf.float32)


reverse_dtype_mapping = {
    bool: 'bool', np.bool_: 'bool', tf.bool: 'bool',
    int: 'int', np.int32: 'int', tf.int32: 'int',
    np.int64: 'long', tf.int64: 'long',
    float: 'float', np.float32: 'float', tf.float32: 'float'
}


def tf_dtype(dtype):
    """Translates dtype specifications in configurations to tensorflow data types.

       Args:
           dtype: String describing a numerical type (e.g. 'float'), numpy data type,
               or numerical type primitive.

       Returns: TensorFlow data type

    """
    if dtype in tf_dtype_mapping:
        return tf_dtype_mapping[dtype]
    else:
        raise TensorforceError.value(name='dtype', value=dtype)


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
    'states', 'actions', 'state', 'action', 'terminal', 'reward', 'deterministic', 'optimization',
    # Types
    'bool', 'int', 'long', 'float',
    # Value specification attributes
    'shape', 'type', 'num_values', 'min_value', 'max_value'
    # Special values?
    'equal', 'loss', 'same', 'x', '*'
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


def is_valid_type(dtype):
    return dtype in ('bool', 'int', 'long', 'float') or dtype in reverse_dtype_mapping


def is_valid_value_type(value_type):
    return value_type in ('state', 'action', 'tensor')


def is_atomic_values_spec(values_spec):
    return 'type' in values_spec or 'shape' in values_spec


def valid_values_spec(values_spec, value_type='tensor', return_normalized=False):
    if not is_valid_value_type(value_type=value_type):
        raise TensorforceError.value(name='value_type', value=value_type)

    if is_atomic_values_spec(values_spec=values_spec):
        value_spec = valid_value_spec(
            value_spec=values_spec, value_type=value_type, return_normalized=return_normalized
        )
        return OrderedDict([(value_type, value_spec)])

    if return_normalized:
        normalized_spec = OrderedDict()

    for name in sorted(values_spec):
        if not is_valid_name(name=name):
            raise TensorforceError.value(name=(value_type + ' name'), value=name)

        result = valid_values_spec(
            values_spec=values_spec[name], value_type=value_type,
            return_normalized=return_normalized
        )
        if return_normalized:
            if len(result) == 1 and next(iter(result)) == value_type:
                normalized_spec[name] = result[value_type]
            else:
                for suffix, spec in result.items():
                    if suffix == value_type:
                        normalized_spec[name] = spec
                    else:
                        normalized_spec[join_scopes(name, suffix)] = spec

    if return_normalized:
        return normalized_spec
    else:
        return True


def valid_value_spec(
    value_spec, value_type='tensor', accept_underspecified=False, return_normalized=False
):
    if not is_valid_value_type(value_type=value_type):
        raise TensorforceError.value(name='value_type', value=value_type)

    value_spec = dict(value_spec)

    if return_normalized:
        normalized_spec = dict()

    if value_type == 'state' and return_normalized:
        dtype = value_spec.pop('type', 'float')
    else:
        dtype = value_spec.pop('type')
    if accept_underspecified and dtype is None:
        if return_normalized:
            normalized_spec['type'] = None
    elif accept_underspecified and is_iterable(x=dtype):
        if not all(is_valid_type(dtype=x) for x in dtype):
            raise TensorforceError.value(name=(value_type + ' spec'), argument='type', value=dtype)
        if return_normalized:
            normalized_spec['type'] = tuple(reverse_dtype_mapping.get(x, x) for x in dtype)
    else:
        if not is_valid_type(dtype=dtype):
            raise TensorforceError.value(name=(value_type + ' spec'), argument='type', value=dtype)
        if return_normalized:
            normalized_spec['type'] = reverse_dtype_mapping.get(dtype, dtype)

    if value_type == 'action' and return_normalized:
        shape = value_spec.pop('shape', ())
    else:
        shape = value_spec.pop('shape')
    if accept_underspecified and shape is None:
        if return_normalized:
            normalized_spec['shape'] = None
    elif is_iterable(x=shape):
        start = int(accept_underspecified and len(shape) > 0 and shape[0] is None)
        if not all(isinstance(dims, int) for dims in shape[start:]):
            raise TensorforceError.value(
                name=(value_type + ' spec'), argument='shape', value=shape
            )
        if accept_underspecified:
            if not all(dims >= -1 for dims in shape[start:]):
                raise TensorforceError.value(
                    name=(value_type + ' spec'), argument='shape', value=shape
                )
        else:
            if not all(dims > 0 or dims == -1 for dims in shape):
                raise TensorforceError.value(
                    name=(value_type + ' spec'), argument='shape', value=shape
                )
        if return_normalized:
            normalized_spec['shape'] = tuple(shape)
    elif return_normalized:
        if not isinstance(shape, int):
            raise TensorforceError.type(name=(value_type + ' spec'), argument='shape', value=shape)
        if accept_underspecified:
            if shape < -1:
                raise TensorforceError.value(
                    name=(value_type + ' spec'), argument='shape', value=shape
                )
        else:
            if not (shape > 0 or shape == -1):
                raise TensorforceError.value(
                    name=(value_type + ' spec'), argument='shape', value=shape
                )
        if return_normalized:
            normalized_spec['shape'] = (shape,)

    if value_type == 'tensor':
        if 'batched' in value_spec:
            batched = value_spec.pop('batched')
            if not isinstance(batched, bool):
                raise TensorforceError.type(
                    name=(value_type + ' spec'), argument='batched', value=batched
                )
            if return_normalized:
                normalized_spec['batched'] = batched

    if dtype == 'bool' or (accept_underspecified and dtype is not None and 'bool' in dtype):
        pass

    if dtype == 'int' or (accept_underspecified and dtype is not None and 'int' in dtype):
        if 'num_values' in value_spec or value_type in ('state', 'action'):
            if 'num_values' not in value_spec:
                raise TensorforceError.required(name=(value_type + ' spec'), value='num_values')
            num_values = value_spec.pop('num_values')
            if isinstance(num_values, (np.int32, np.int64)):
                num_values = num_values.item()
            if not isinstance(num_values, int):
                raise TensorforceError.type(
                    name=(value_type + ' spec'), argument='num_values', value=num_values
                )
            if accept_underspecified:
                if not (num_values > 1 or num_values == 0):
                    raise TensorforceError.value(
                        name=(value_type + ' spec'), argument='num_values', value=num_values
                    )
            else:
                if num_values <= 1:
                    raise TensorforceError.value(
                        name=(value_type + ' spec'), argument='num_values', value=num_values
                    )
            if return_normalized:
                normalized_spec['num_values'] = num_values

    if dtype == 'long' or (accept_underspecified and dtype is not None and 'long' in dtype):
        pass

    if dtype == 'float' or (accept_underspecified and dtype is not None and 'float' in dtype):
        if 'min_value' in value_spec:
            min_value = value_spec.pop('min_value')
            max_value = value_spec.pop('max_value')
            if isinstance(min_value, np_dtype(dtype='float')):
                min_value = min_value.item()
            if isinstance(max_value, np_dtype(dtype='float')):
                max_value = max_value.item()
            if not isinstance(min_value, float):
                raise TensorforceError.type(
                    name=(value_type + ' spec'), argument='min_value', value=min_value
                )
            if not isinstance(max_value, float):
                raise TensorforceError.type(
                    name=(value_type + ' spec'), argument='max_value', value=max_value
                )
            if min_value >= max_value:
                raise TensorforceError.value(
                    name=(value_type + ' spec'), argument='min/max_value',
                    value=(min_value, max_value)
                )
            if return_normalized:
                normalized_spec['min_value'] = min_value
                normalized_spec['max_value'] = max_value

    if len(value_spec) > 0:
        raise TensorforceError.value(name=(value_type + ' spec'), value=tuple(value_spec))

    if return_normalized:
        return normalized_spec
    else:
        return True


def is_value_spec_more_specific(specific_value_spec, value_spec):
    # Check type consistency
    specific_dtype = specific_value_spec['type']
    dtype = value_spec['type']
    if dtype is None:
        pass
    elif is_iterable(x=dtype):
        if is_iterable(x=specific_dtype):
            if not all(x in dtype for x in specific_dtype):
                return False
        elif specific_dtype not in dtype:
            return False
    elif specific_dtype != dtype:
        return False

    # Check shape consistency
    specific_shape = specific_value_spec['shape']
    shape = value_spec['shape']
    if shape is None:
        pass
    elif len(shape) > 0 and shape[0] is None:
        if len(specific_shape) < len(shape) - 1:
            return False
        elif not all(
            a == b or b == 0 for a, b in zip(specific_shape[-len(shape) + 1:], shape[1:])
        ):
            return False
    elif len(specific_shape) > 0 and specific_shape[0] is None:
        if len(specific_shape) - 1 > len(shape):
            return False
        elif not all(
            a == b or b == 0 for a, b in zip(specific_shape[1:], shape[-len(specific_shape) + 1:])
        ):
            return False
    else:
        if len(specific_shape) != len(shape):
            return False
        elif not all(a == b or b == 0 for a, b in zip(specific_shape, shape)):
            return False

    # Check batched consistency

    # Check num_values consistency
    specific_num_values = specific_value_spec.get('num_values', 0)
    num_values = value_spec.get('num_values', 0)
    if not (specific_num_values == num_values or num_values == 0):
        return False

    # Check min/max_value consistency
    if 'min_value' not in specific_value_spec and 'min_value' in value_spec:
        return False

    return True


def unify_value_specs(value_spec1, value_spec2):
    if not valid_value_spec(value_spec=value_spec1, accept_underspecified=True):
        raise TensorforceError.value(
            name='unify_value_spec', argument='value_spec1', value=value_spec1
        )
    elif not valid_value_spec(value_spec=value_spec2, accept_underspecified=True):
        raise TensorforceError.value(
            name='unify_value_spec', argument='value_spec2', value=value_spec2
        )

    unified_value_spec = dict()

    # Unify type
    dtype1 = value_spec1['type']
    dtype2 = value_spec2['type']
    if dtype1 is None:
        dtype = dtype2
    elif dtype2 is None:
        dtype = dtype1
    elif is_iterable(x=dtype1):
        if is_iterable(x=dtype2):
            if all(x in dtype1 for x in dtype2):
                dtype = dtype2
            elif all(x in dtype2 for x in dtype1):
                dtype = dtype1
            else:
                raise TensorforceError.mismatch(
                    name='value-spec', argument='type', value1=dtype1, value2=dtype2
                )
        elif dtype2 in dtype1:
            dtype = dtype2
        else:
            raise TensorforceError.mismatch(
                name='value-spec', argument='type', value1=dtype1, value2=dtype2
            )
    elif is_iterable(x=dtype2):
        if dtype1 in dtype2:
            dtype = dtype1
        else:
            raise TensorforceError.mismatch(
                name='value-spec', argument='type', value1=dtype1, value2=dtype2
            )
    elif dtype1 == dtype2:
        dtype = dtype1
    else:
        raise TensorforceError.mismatch(
            name='value-spec', argument='type', value1=dtype1, value2=dtype2
        )
    unified_value_spec['type'] = dtype

    # Unify shape
    shape1 = value_spec1['shape']
    shape2 = value_spec2['shape']
    if shape1 is None:
        shape = shape2
    elif shape2 is None:
        shape = shape1
    else:
        reverse_shape = list()
        for n in range(max(len(shape1), len(shape2))):
            if len(shape1) <= n:
                if shape2[-n - 1] is not None:
                    reverse_shape.append(shape2[-n - 1])
            elif len(shape2) <= n:
                if shape1[-n - 1] is not None:
                    reverse_shape.append(shape1[-n - 1])
                reverse_shape.append(shape1[-n - 1])
            elif shape1[-n - 1] is None:
                reverse_shape.append(shape2[-n - 1])
            elif shape2[-n - 1] is None:
                reverse_shape.append(shape1[-n - 1])
            elif shape1[-n - 1] == 0:
                reverse_shape.append(shape2[-n - 1])
            elif shape2[-n - 1] == 0:
                reverse_shape.append(shape1[-n - 1])
            elif shape1[-n - 1] == -1:
                reverse_shape.append(shape2[-n - 1])
            elif shape2[-n - 1] == -1:
                reverse_shape.append(shape1[-n - 1])
            elif shape1[-n - 1] == shape2[-n - 1]:
                reverse_shape.append(shape1[-n - 1])
            else:
                raise TensorforceError.mismatch(
                    name='value-spec', argument='shape', value1=shape1, value2=shape2
                )
        shape = tuple(reversed(reverse_shape))
    unified_value_spec['shape'] = shape

    # # Unify batched
    # if 'batched' in value_spec1 or 'batched' in value_spec2:
    #     batched1 = value_spec1.get('batched', False)
    #     batched2 = value_spec2.get('batched', False)
    #     if batched1 is batched2:
    #         batched = batched1
    #     else:
    #         raise TensorforceError.mismatch(
    #             name='value-spec', argument='batched', value1=batched1, value2=batched2
    #         )
    #     unified_value_spec['batched'] = batched

    # Unify num_values
    if 'num_values' in value_spec1 and 'num_values' in value_spec2:
        num_values1 = value_spec1['num_values']
        num_values2 = value_spec2['num_values']
        if num_values1 == 0:
            num_values = num_values2
        elif num_values2 == 0:
            num_values = num_values1
        elif num_values1 == num_values2:
            # num_values = max(num_values1, num_values2)
            num_values = num_values1
        else:
            raise TensorforceError.mismatch(
                name='value-spec', argument='num_values', value1=num_values1, value2=num_values2
            )
        unified_value_spec['num_values'] = num_values

    # Unify min/max_value
    if 'min_value' in value_spec1 and 'min_value' in value_spec2:
        min_value = min(value_spec1['min_value'], value_spec2['min_value'])
        max_value = max(value_spec1['max_value'], value_spec2['max_value'])
        unified_value_spec['min_value'] = min_value
        unified_value_spec['max_value'] = max_value

    return unified_value_spec


def is_consistent_with_value_spec(value_spec, x):
    if value_spec['type'] is None:
        pass
    elif is_iterable(x=value_spec['type']) and dtype(x=x) in value_spec['type']:
        pass
    elif dtype(x=x) == value_spec['type']:
        pass
    else:
        return False
    if value_spec['shape'] is None:
        pass
    elif len(shape(x=x)) != len(value_spec['shape']) + int(value_spec.get('batched', True)):
        return False
    elif value_spec.get('batched', True):
        if not all(
            a == b or b == 0 or b == -1 for a, b in zip(shape(x=x), (-1,) + value_spec['shape'])
        ):
            return False
    elif not all(a == b or b == 0 or b == -1 for a, b in zip(shape(x=x), value_spec['shape'])):
        return False
    # num_values
    # min/max_value
    return True


def normalize_values(value_type, values, values_spec):
    if not is_valid_value_type(value_type=value_type):
        raise TensorforceError.value(name='value_type', value=value_type)

    if len(values_spec) == 1 and next(iter(values_spec)) == value_type:
        # Spec defines only a single value
        if isinstance(values, dict):
            if len(values) != 1 or value_type not in values:
                TensorforceError.value(name=(value_type + ' spec'), value=values)
            return values

        else:
            return OrderedDict([(value_type, values)])

    normalized_values = OrderedDict()
    for normalized_name in values_spec:
        value = values
        for name in normalized_name.split('/'):
            value = value[name]
        normalized_values[normalized_name] = value

        # Check whether only expected values present!

    return normalized_values


def unpack_values(value_type, values, values_spec):
    if not is_valid_value_type(value_type=value_type):
        raise TensorforceError.value(name='value_type', value=value_type)

    if len(values_spec) == 1 and next(iter(values_spec)) == value_type:
        # Spec defines only a single value
        return values[value_type]

    unpacked_values = dict()
    for normalized_name in values_spec:
        unpacked_value = unpacked_values
        names = normalized_name.split('/')
        for name in names[:-1]:
            if name not in unpacked_value:
                unpacked_value[name] = dict()
            unpacked_value = unpacked_value[name]
        unpacked_value[names[-1]] = values.pop(normalized_name)

    if len(values) > 0:
        raise TensorforceError.unexpected()

    return unpacked_values


# def get_object(obj, predefined_objects=None, default_object=None, kwargs=None):
#     """
#     Utility method to map some kind of object specification to its content,
#     e.g. optimizer or baseline specifications to the respective classes.

#     Args:
#         obj: A specification dict (value for key 'type' optionally specifies
#                 the object, options as follows), a module path (e.g.,
#                 my_module.MyClass), a key in predefined_objects, or a callable
#                 (e.g., the class type object).
#         predefined_objects: Dict containing predefined set of objects,
#                 accessible via their key
#         default_object: Default object is no other is specified
#         kwargs: Arguments for object creation

#     Returns: The retrieved object

#     """
#     args = ()
#     kwargs = dict() if kwargs is None else kwargs

#     if isinstance(obj, str) and os.path.isfile(obj):
#         with open(obj, 'r') as fp:
#             obj = json.load(fp=fp)
#     if isinstance(obj, dict):
#         kwargs.update(obj)
#         obj = kwargs.pop('type', None)

#     if predefined_objects is not None and obj in predefined_objects:
#         obj = predefined_objects[obj]
#     elif isinstance(obj, str):
#         if obj.find('.') != -1:
#             module_name, function_name = obj.rsplit('.', 1)
#             module = importlib.import_module(module_name)
#             obj = getattr(module, function_name)
#         else:
#             raise TensorforceError("Error: object {} not found in predefined objects: {}".format(
#                 obj,
#                 list(predefined_objects or ())
#             ))
#     elif callable(obj):
#         pass
#     elif default_object is not None:
#         args = (obj,)
#         obj = default_object
#     else:
#         # assumes the object is already instantiated
#         return obj

#     return obj(*args, **kwargs)


# def prepare_kwargs(raw, string_parameter='name'):
#     """
#     Utility method to convert raw string/diction input into a dictionary to pass
#     into a function.  Always returns a dictionary.

#     Args:
#         raw: string or dictionary, string is assumed to be the name of the activation
#                 activation function.  Dictionary will be passed through unchanged.

#     Returns: kwargs dictionary for **kwargs

#     """
#     kwargs = dict()

#     if isinstance(raw, dict):
#         kwargs.update(raw)
#     elif isinstance(raw, str):
#         kwargs[string_parameter] = raw

#     return kwargs


def strip_name_scope(name, base_scope):
    if name.startswith(base_scope):
        return name[len(base_scope):]
    else:
        return name


class SavableComponent(object):
    """
    Component that can save and restore its own state.
    """

    def register_saver_ops(self):
        """
        Registers the saver operations to the graph in context.
        """

        variables = self.get_savable_variables()
        if variables is None or len(variables) == 0:
            self._saver = None
            return

        base_scope = self._get_base_variable_scope()
        variables_map = {strip_name_scope(v.name, base_scope): v for v in variables}

        self._saver = tf.train.Saver(
            var_list=variables_map,
            reshape=False,
            sharded=False,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=True,
            pad_step_number=False,
            save_relative_paths=True
        )

    def get_savable_variables(self):
        """
        Returns the list of all the variables this component is responsible to save and restore.

        Returns:
            The list of variables that will be saved or restored.
        """

        raise NotImplementedError()

    def save(self, sess, save_path, timestep=None):
        """
        Saves this component's managed variables.

        Args:
            sess: The session for which to save the managed variables.
            save_path: The path to save data to.
            timestep: Optional, the timestep to append to the file name.

        Returns:
            Checkpoint path where the model was saved.
        """

        if self._saver is None:
            raise TensorforceError("register_saver_ops should be called before save")
        return self._saver.save(
            sess=sess,
            save_path=save_path,
            global_step=timestep,
            write_meta_graph=False,
            write_state=True,  # Do we need this?
        )

    def restore(self, sess, save_path):
        """
        Restores the values of the managed variables from disk location.

        Args:
            sess: The session for which to save the managed variables.
            save_path: The path used to save the data to.
        """

        if self._saver is None:
            raise TensorforceError("register_saver_ops should be called before restore")
        self._saver.restore(sess=sess, save_path=save_path)

    def _get_base_variable_scope(self):
        """
        Returns the portion of the variable scope that is considered a base for this component. The variables will be
        saved with names relative to that scope.

        Returns:
            The name of the base variable scope, should always end with "/".
        """

        raise NotImplementedError()
