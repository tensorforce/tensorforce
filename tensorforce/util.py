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
import numpy as np
import tensorflow as tf


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def cumulative_discount(rewards, terminals, discount):
    if discount == 0.0:
        return rewards
    cumulative = 0.0
    for n, (reward, terminal) in reversed(list(enumerate(zip(rewards, terminals)))):
        if terminal:
            cumulative = 0.0
        cumulative = reward + cumulative * discount
        rewards[n] = cumulative
    return rewards


def np_dtype(dtype):
    if dtype is None or dtype == 'float':
        return np.float32
    elif dtype == 'int':
        return np.int32
    elif dtype == 'bool':
        return np._bool
    else:
        raise Exception()


def tf_dtype(dtype):
    if dtype is None or dtype == 'float':
        return tf.float32
    elif dtype == 'int':
        return tf.int32
    else:
        raise Exception()


def function(f):
    module_name, function_name = f.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)



# def make_function(data, fk):
#     """
#     Take data dict and convert string function reference with key `fk` to a real function reference, using
#     `fk`_args as *args and `fk`_kwargs as **kwargs, removing these keys from the data dict.

#     :param data: data dict
#     :param fk: string function key
#     :return: boolean
#     """
#     fn = data.get(fk)

#     if fn is None:
#         return True
#     elif callable(fn):
#         return True
#     else:
#         args_val = "{}_args".format(fk)
#         kwargs_val = "{}_kwargs".format(fk)

#         args = data.pop(args_val, None)
#         kwargs = data.pop(kwargs_val, None)

#         func = get_function(fn)

#         if args is None and kwargs is None:
#             # If there are no args and no kwargs, just return the function reference
#             data[fk] = func
#             return True

#         # Otherwise, call the function
#         if args is None:
#             args = []
#         if kwargs is None:
#             kwargs = {}

#         data[fk] = func(*args, **kwargs)
#         return True
