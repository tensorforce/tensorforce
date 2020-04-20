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

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from tensorforce.core.utils import NestedDict


class ArrayDict(NestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=np.ndarray, overwrite=False, **kwargs)


class ModuleDict(NestedDict):

    def __init__(self, *args, **kwargs):
        from tensorforce.core import Module
        super().__init__(*args, value_type=Module, overwrite=False, **kwargs)


class SignatureDict(NestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=tf.TensorSpec, overwrite=False, **kwargs)

    def num_args(self):
        return super(NestedDict, self).__len__()

    def to_list(self):
        return [
            spec.to_list() if isinstance(spec, self.__class__) else spec
            for spec in super(NestedDict, self).values()
        ]

    def kwargs_to_args(self, kwargs):
        args = list()
        for name, spec in super(NestedDict, self).items():
            arg = kwargs[name]
            if isinstance(spec, self.__class__):
                assert isinstance(arg, TensorDict)
                args.append(spec.kwargs_to_args(kwargs=arg))
            else:
                assert isinstance(arg, tf.Tensor)
                args.append(arg)
        return args

    def args_to_kwargs(self, args):
        kwargs = OrderedDict()
        for (name, spec), arg in zip(super(NestedDict, self).items(), args):
            if isinstance(spec, self.__class__):
                assert isinstance(arg, list)
                kwargs[name] = TensorDict(spec.args_to_kwargs(args=arg))
            else:
                assert isinstance(arg, tf.Tensor)
                kwargs[name] = arg
        return kwargs


class TensorDict(NestedDict):

    def __init__(self, *args, overwrite=True, **kwargs):
        super().__init__(*args, value_type=tf.Tensor, overwrite=overwrite, **kwargs)

    def to_list(self):
        return [
            value.to_list() if isinstance(value, TensorDict) else value
            for value in super(NestedDict, self).values()
        ]

    def from_list(self, xs):
        return TensorDict((
            (name, value.from_list(xs=x) if isinstance(value, TensorDict) else x)
            for (name, value), x in zip(super(NestedDict, self).items(), xs)
        ))


class VariableDict(NestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=tf.Variable, overwrite=False, **kwargs)
