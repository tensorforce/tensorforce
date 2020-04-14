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

from tensorforce import TensorforceError, util
from tensorforce.core.utils import NestedDict, TensorDict, TensorSpec


class TensorsSpec(NestedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, value_type=TensorSpec, overwrite=False, **kwargs)

    def signature(self, batched):
        return [spec.signature(batched=batched) for spec in super(NestedDict, self).values()]

    def tf_assert(self, x, batch_size=None, include_type_shape=False, message=None):
        if not isinstance(x, TensorDict):
            raise TensorforceError.type(name='TensorsSpec.tf_assert', argument='x', dtype=type(x))

        assertions = list()
        for name, spec, x in util.zip_items(self, x):
            assertions.extend(spec.tf_assert(
                x=x, batch_size=batch_size, include_type_shape=include_type_shape,
                message=(None if message is None else message.format(name=name, issue='{issue}'))
            ))

        return assertions

    def to_tensor(self, value, batched):
        if not isinstance(value, dict):
            raise TensorforceError.type(
                name='TensorsSpec.to_tensor', argument='value', dtype=type(value)
            )

        # TODO: improve exception message to include invalid keys
        if set(value) != set(super().__iter__()):
            raise TensorforceError.value(
                name='TensorsSpec.to_tensor', argument='value', value=value
            )

        tensor = TensorDict()
        for name, spec in super().items():
            tensor[name] = spec.to_tensor(value=value[name], batched=batched)

        return tensor

    def from_tensor(self, tensor, batched):
        if not isinstance(tensor, TensorDict):
            raise TensorforceError.type(
                name='TensorsSpec.from_tensor', argument='tensor', dtype=type(tensor)
            )

        # TODO: improve exception message to include invalid keys
        if set(tensor) != set(super().__iter__()):
            raise TensorforceError.value(
                name='TensorsSpec.from_tensor', argument='tensor', value=tensor
            )

        value = OrderedDict()
        for name, spec in super().items():
            value[name] = spec.from_tensor(tensor=tensor[name], batched=batched)

        return value

    def __setitem__(self, key, value):
        if not isinstance(value, TensorSpec) and not isinstance(value, TensorsSpec):
            if not isinstance(value, dict):
                raise TensorforceError.type(name='TensorsSpec', argument='value', dtype=type(value))
            elif 'type' in value or 'shape' in value:
                value = TensorSpec(**value, overwrite=self.overwrite)
            else:
                value = TensorsSpec(value)

        if key == 'horizons':
            if not isinstance(value, TensorSpec) or value.type != 'int' or value.shape != (2,):
                raise TensorforceError.value(name='TensorsSpec', argument='horizons', value=value)

        elif key == 'parallel':
            if not isinstance(value, TensorSpec) or value.type != 'int' or value.shape != ():
                raise TensorforceError.value(name='TensorsSpec', argument='parallel', value=value)

        elif key == 'reward':
            if not isinstance(value, TensorSpec) or value.type != 'float' or value.shape != ():
                raise TensorforceError.value(name='TensorsSpec', argument='reward', value=value)

        elif key == 'terminal':
            if not isinstance(value, TensorSpec) or value.type != 'int' or value.shape != ():
                raise TensorforceError.value(name='TensorsSpec', argument='terminal', value=value)

        super().__setitem__(key, value)
