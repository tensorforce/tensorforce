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
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.training.tracking.data_structures import sticky_attribute_assignment

from tensorforce.core.utils import NestedDict


class TrackableNestedDict(NestedDict, AutoTrackable):

    def __init__(self, arg=None, *, value_type=None, overwrite=None, **kwargs):
        self._maybe_initialize_trackable()
        super().__init__(
            arg=arg, value_type=value_type, overwrite=overwrite, singleton=None, **kwargs
        )

    def __setattr__(self, name, value):
        if name.startswith('_self_'):
            super(NestedDict, self).__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __setitem__(self, key, value):
        if key is None:
            value = sticky_attribute_assignment(
                trackable=self, value=value, name=self.__class__._SINGLETON
            )
        else:
            value = sticky_attribute_assignment(trackable=self, value=value, name=key)
        super().__setitem__(key, value)

    # def __iter__(self):
    #     for name in super().__iter__():
    #         if name is None:
    #             yield
    #         else:
    #             yield name

    # def items(self):
    #     for name, value in super().items():
    #         if name is None:
    #             yield self.__class__._SINGLETON, value
    #         else:
    #             yield name, value


class ArrayDict(NestedDict):

    def __init__(self, *args, singleton=None, **kwargs):
        super().__init__(
            *args, value_type=np.ndarray, overwrite=False, singleton=singleton, **kwargs
        )

    def __setitem__(self, key, value):
        if not isinstance(value, dict):
            value = np.asarray(value)
        super().__setitem__(key, value)

    def to_kwargs(self):
        if self.is_singleton():
            value = self.singleton()
            if isinstance(value, self.value_type):
                return value
            else:
                return value.to_kwargs()
        else:
            return OrderedDict(((name, arg) for name, arg in super(NestedDict, self).items()))


class ListDict(NestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=list, overwrite=False, singleton=None, **kwargs)


class ModuleDict(TrackableNestedDict):

    def __init__(self, *args, **kwargs):
        from tensorforce.core import Module
        super().__init__(*args, value_type=Module, overwrite=False, **kwargs)


class SignatureDict(NestedDict):

    def __init__(self, *args, singleton=None, **kwargs):
        super().__init__(
            *args, value_type=tf.TensorSpec, overwrite=False, singleton=singleton, **kwargs
        )

    def num_args(self):
        return super(NestedDict, self).__len__()

    def to_list(self):
        if self.is_singleton():
            spec = self.singleton()
            if isinstance(spec, self.value_type):
                return spec
            else:
                return spec.to_list()

        else:
            return [
                spec if isinstance(spec, self.value_type) else spec.to_list()
                for spec in super(NestedDict, self).values()
                if isinstance(spec, self.value_type) or len(spec) > 0
            ]

    def kwargs_to_args(self, *, kwargs, is_outer=True, flatten=False):
        if self.is_singleton():
            spec = self.singleton()
            if isinstance(spec, self.value_type):
                if isinstance(kwargs, (tf.IndexedSlices, tf.Tensor, tf.Variable)):
                    # Special case: API input arguments are raw values, not singleton dicts
                    return kwargs
                else:
                    assert isinstance(kwargs, TensorDict) and kwargs.is_singleton()
                    arg = kwargs.singleton()
                    assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable)), (self, kwargs)
                    return arg
            else:
                return spec.kwargs_to_args(kwargs=kwargs, is_outer=False, flatten=flatten)

        else:
            if is_outer:
                assert isinstance(kwargs, (dict, list, tuple))
            else:
                assert isinstance(kwargs, TensorDict), (self, kwargs)
            args = list()
            for index, (name, spec) in enumerate(super(NestedDict, self).items()):
                if is_outer and isinstance(kwargs, (list, tuple)):
                    if index < len(kwargs):
                        arg = kwargs[index]
                elif name in kwargs:
                    arg = kwargs[name]
                if isinstance(spec, self.value_type):
                    assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable))
                    args.append(arg)
                elif len(spec) == 0:
                    continue
                else:
                    arg = spec.kwargs_to_args(kwargs=arg, is_outer=False, flatten=flatten)
                    if flatten and isinstance(arg, tuple):
                        args.extend(arg)
                    else:
                        args.append(arg)
            return tuple(args)

    def args_to_kwargs(self, *, args, is_outer=True, outer_tuple=False, flattened=False):
        if flattened is True and is_outer and isinstance(args, tuple):
            args = list(args)

        if self.is_singleton():
            spec = self.singleton()
            if isinstance(spec, self.value_type):
                if flattened and isinstance(args, list):
                    args = args.pop(0)
                assert isinstance(args, (tf.IndexedSlices, tf.Tensor, tf.Variable)), (self, args)
                kwargs = args
            else:
                kwargs = spec.args_to_kwargs(args=args, is_outer=False, flattened=flattened)
            if is_outer and outer_tuple:
                return kwargs
            else:
                return TensorDict(singleton=kwargs)

        else:
            assert isinstance(args, (list, tuple)), (self, args)
            kwargs = TensorDict()
            index = 0
            for name, spec in super(NestedDict, self).items():
                if not flattened and index < len(args):
                    arg = args[index]
                else:
                    arg = None
                if isinstance(spec, self.value_type):
                    if flattened and isinstance(args, list):
                        arg = args.pop(0)
                    assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable))
                    kwargs[name] = arg
                    index += 1
                elif len(spec) == 0:
                    # Recover empty arguments
                    # (False incompatible with TensorDict, so ensures it is never called)
                    kwargs[name] = spec.fmap(function=(lambda: False), cls=TensorDict)
                else:
                    if flattened:
                        arg = args
                    kwargs[name] = spec.args_to_kwargs(args=arg, is_outer=False, flattened=flattened)
                    index += 1
            if is_outer and outer_tuple:
                return tuple(super(NestedDict, kwargs).values())
            else:
                return kwargs


class TensorDict(NestedDict):

    def __init__(self, *args, overwrite=True, singleton=None, **kwargs):
        # TensorSpec required for SavedModel (presumably for spec tracing)
        super().__init__(
            *args, value_type=(
                tf.IndexedSlices, tf.IndexedSlicesSpec, tf.Tensor, tf.TensorSpec, tf.Variable
            ), overwrite=overwrite, singleton=singleton, **kwargs
        )

    def to_kwargs(self):
        if self.is_singleton():
            value = self.singleton()
            if isinstance(value. self.value_type):
                return value
            else:
                return value.to_kwargs()
        else:
            return OrderedDict(((name, arg) for name, arg in super(NestedDict, self).items()))


class VariableDict(TrackableNestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=tf.Variable, overwrite=False, **kwargs)
