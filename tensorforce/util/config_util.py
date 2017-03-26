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

"""
Utility functions concerning configurations
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six import callable
import importlib


def get_function(fn):
    """
    Get function reference by full module path.

    :param fn: Callable object or String containing the full function path
    :return: Function reference
    """
    if callable(fn):
        return fn
    else:
        module_name, function_name = fn.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)


def make_function(data, fk):
    """
    Take data dict and convert string function reference with key `fk` to a real function reference, using
    `fk`_args as *args and `fk`_kwargs as **kwargs, removing these keys from the data dict.

    :param data: data dict
    :param fk: string function key
    :return: boolean
    """
    fn = data.get(fk)

    if fn is None:
        return True
    elif callable(fn):
        return True
    else:
        args_val = "{}_args".format(fk)
        kwargs_val = "{}_kwargs".format(fk)

        args = data.pop(args_val, None)
        kwargs = data.pop(kwargs_val, None)

        func = get_function(fn)

        if args is None and kwargs is None:
            # If there are no args and no kwargs, just return the function reference
            data[fk] = func
            return True

        # Otherwise, call the function
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        data[fk] = func(*args, **kwargs)
        return True
