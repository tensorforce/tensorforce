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
        super().__init__(arg=arg, value_type=value_type, overwrite=overwrite, **kwargs)

    def __setattr__(self, name, value):
        if name.startswith('_self_'):
            super(NestedDict, self).__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __setitem__(self, key, value):
        value = sticky_attribute_assignment(trackable=self, value=value, name=key)
        super().__setitem__(key, value)


class ArrayDict(NestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=np.ndarray, overwrite=False, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, dict):
            value = np.asarray(value)
        super().__setitem__(key, value)


class ListDict(NestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=list, overwrite=False, **kwargs)


class ModuleDict(TrackableNestedDict):

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

    def kwargs_to_args(self, *, kwargs, is_outer_args=False):
        args = list()
        for index, (name, spec) in enumerate(super(NestedDict, self).items()):
            if is_outer_args and isinstance(kwargs, (list, tuple)):
                arg = kwargs[index]
            else:
                arg = kwargs.get(name, TensorDict())
            if isinstance(spec, self.__class__):
                assert isinstance(arg, TensorDict)
                args.append(spec.kwargs_to_args(kwargs=arg))
            else:
                assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable))
                args.append(arg)
        return args

    def args_to_kwargs(self, *, args):
        kwargs = TensorDict()
        index = 0
        for name, spec in super(NestedDict, self).items():
            if index < len(args):
                arg = args[index]
            else:
                arg = None
            if isinstance(spec, tf.TensorSpec):
                assert isinstance(arg, (tf.IndexedSlices, tf.Tensor, tf.Variable))
                kwargs[name] = arg
                index += 1
            else:
                if isinstance(arg, (list, tuple)) and super(NestedDict, spec).__len__() == len(arg):
                    kwargs[name] = spec.args_to_kwargs(args=arg)
                    index += 1
                else:
                    kwargs[name] = spec.args_to_kwargs(args=())
        return kwargs


class TensorDict(NestedDict):

    def __init__(self, *args, overwrite=True, **kwargs):
        # TensorSpec required for SavedModel (presumably for spec tracing)
        super().__init__(
            *args, value_type=(tf.IndexedSlices, tf.Tensor, tf.TensorSpec, tf.Variable),
            overwrite=overwrite, **kwargs
        )

    def to_kwargs(self):
        return OrderedDict(((name, arg) for name, arg in super(NestedDict, self).items()))


class VariableDict(TrackableNestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=tf.Variable, overwrite=False, **kwargs)
