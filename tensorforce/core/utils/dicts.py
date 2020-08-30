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

    def to_dict(self):
        if self.is_singleton():
            value = self.singleton()
            if isinstance(value, self.value_type):
                return value
            else:
                return value.to_dict()
        else:
            return OrderedDict((
                (name, arg) if isinstance(arg, self.value_type) else (name, arg.to_dict())
                for name, arg in super(NestedDict, self).items()
            ))

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

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key is None or key == self.__class__._SINGLETON or self.is_singleton() or '/' in key or \
                not isinstance(value, self.value_type):
            pass
        elif value._name is None:
            value._name = key
        else:
            assert value._name == key

    def num_args(self):
        return super(NestedDict, self).__len__()

    def to_list(self, to_dict=False):
        if self.is_singleton():
            spec = self.singleton()
            if isinstance(spec, self.value_type):
                return spec
            else:
                return spec.to_list()

        else:
            return [
                spec if isinstance(spec, self.value_type) else (
                    spec.to_dict() if to_dict else spec.to_list()
                ) for spec in super(NestedDict, self).values()
                if isinstance(spec, self.value_type) or len(spec) > 0
            ]

    def to_dict(self):
        if self.is_singleton():
            spec = self.singleton()
            if isinstance(spec, self.value_type):
                return spec
            else:
                return spec.to_dict()

        else:
            return OrderedDict((
                (name, (spec if isinstance(spec, self.value_type) else spec.to_dict()))
                for name, spec in super(NestedDict, self).items()
                if isinstance(spec, self.value_type) or len(spec) > 0
            ))

    def kwargs_to_args(self, *, kwargs, to_dict=False, outer_tuple=False, is_outer=True):
        if self.is_singleton():
            spec = self.singleton()
            if isinstance(spec, self.value_type):
                if isinstance(kwargs, (tf.IndexedSlices, tf.Tensor, tf.Variable)):
                    # Special case: API input arguments are raw values, not singleton dicts
                    assert spec.is_compatible_with(spec_or_tensor=kwargs), (spec, kwargs)
                    return kwargs
                else:
                    assert isinstance(kwargs, TensorDict) and kwargs.is_singleton()
                    arg = kwargs.singleton()
                    assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable))
                    assert spec.is_compatible_with(spec_or_tensor=arg), (spec, arg)
                    return arg
            else:
                return spec.kwargs_to_args(kwargs=kwargs, to_dict=to_dict, is_outer=False)

        else:
            if is_outer:
                assert isinstance(kwargs, (dict, list, tuple))
            else:
                assert isinstance(kwargs, TensorDict), (self, kwargs)
            if to_dict:
                args = dict()
            else:
                args = list()
            for index, (name, spec) in enumerate(super(NestedDict, self).items()):
                if is_outer and isinstance(kwargs, (list, tuple)):
                    if index < len(kwargs):
                        arg = kwargs[index]
                elif name in kwargs:
                    arg = kwargs[name]
                if isinstance(spec, self.value_type):
                    assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable))
                    if isinstance(arg, tf.IndexedSlices):
                        # TODO: why does IndexedSlicesSpec not work?
                        # spec = tf.IndexedSlicesSpec(
                        #     shape=spec.shape, dtype=spec.dtype, indices_dtype=arg.indices.dtype
                        # )
                        # assert spec.is_compatible_with(spec_or_value=arg), (name, spec, arg)
                        assert tf.TensorSpec(
                            shape=((None,) + spec.shape[1:]), dtype=spec.dtype
                        ).is_compatible_with(spec_or_tensor=arg.values)
                        assert tf.TensorSpec(
                            shape=(None,), dtype=arg.indices.dtype
                        ).is_compatible_with(spec_or_tensor=arg.indices)
                    else:
                        assert spec.is_compatible_with(spec_or_tensor=arg), (name, spec, arg)
                    if to_dict:
                        args[name] = arg
                    else:
                        args.append(arg)
                elif len(spec) == 0:
                    continue
                else:
                    arg = spec.kwargs_to_args(kwargs=arg, to_dict=to_dict, is_outer=False)
                    if to_dict:
                        args[name] = arg
                    else:
                        args.append(arg)
            if to_dict:
                if outer_tuple and is_outer:
                    args = tuple(args.values())
            else:
                args = tuple(args)
            return args

    def args_to_kwargs(self, *, args, from_dict=False, outer_tuple=False, is_outer=True):
        if self.is_singleton():
            spec = self.singleton()
            if isinstance(spec, self.value_type):
                assert isinstance(args, (tf.IndexedSlices, tf.Tensor, tf.Variable)), (self, args)
                assert spec.is_compatible_with(spec_or_tensor=args), (spec, args)
                kwargs = args
            else:
                kwargs = spec.args_to_kwargs(args=args, from_dict=from_dict, is_outer=False)
            if outer_tuple and is_outer:
                return kwargs
            else:
                return TensorDict(singleton=kwargs)

        else:
            if is_outer:
                assert isinstance(args, (dict, list, tuple))
            elif from_dict:
                assert isinstance(args, dict), (self, args)
            else:
                assert isinstance(args, (list, tuple)), (self, args)
            kwargs = TensorDict()
            index = 0
            for name, spec in super(NestedDict, self).items():
                if from_dict and isinstance(args, dict):
                    arg = args.get(name)
                elif index < len(args):
                    assert not from_dict or is_outer
                    arg = args[index]
                else:
                    arg = None
                if isinstance(spec, self.value_type):
                    assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable))
                    if isinstance(arg, tf.IndexedSlices):
                        # TODO: why does IndexedSlicesSpec not work?
                        # spec = tf.IndexedSlicesSpec(
                        #     shape=spec.shape, dtype=spec.dtype, indices_dtype=arg.indices.dtype
                        # )
                        # assert spec.is_compatible_with(spec_or_value=arg), (name, spec, arg)
                        assert tf.TensorSpec(
                            shape=((None,) + spec.shape[1:]), dtype=spec.dtype
                        ).is_compatible_with(spec_or_tensor=arg.values)
                        assert tf.TensorSpec(
                            shape=(None,), dtype=arg.indices.dtype
                        ).is_compatible_with(spec_or_tensor=arg.indices)
                    else:
                        assert spec.is_compatible_with(spec_or_tensor=arg), (name, spec, arg)
                    kwargs[name] = arg
                    index += 1
                elif len(spec) == 0:
                    # Recover empty arguments
                    # (False incompatible with TensorDict, so ensures it is never called)
                    kwargs[name] = spec.fmap(function=(lambda: False), cls=TensorDict)
                else:
                    kwargs[name] = spec.args_to_kwargs(
                        args=arg, from_dict=from_dict, is_outer=False
                    )
                    index += 1
            if outer_tuple and is_outer:
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
