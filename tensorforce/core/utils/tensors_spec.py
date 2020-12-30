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

from tensorforce import TensorforceError
from tensorforce.core.utils import ArrayDict, NestedDict, SignatureDict, TensorDict, TensorSpec


class TensorsSpec(NestedDict):

    def __init__(self, *args, singleton=None, **kwargs):
        super().__init__(
            *args, value_type=TensorSpec, overwrite=False, singleton=singleton, **kwargs
        )

    def signature(self, *, batched):
        return self.fmap(function=(lambda spec: spec.signature(batched=batched)), cls=SignatureDict)

    def to_tensor(self, *, value, batched, recover_empty=False, name='TensorSpec.to_tensor'):
        if value is not None and not isinstance(value, ArrayDict):
            raise TensorforceError.type(name=name, argument='value', dtype=type(value))

        # TODO: improve exception message to include invalid keys
        if not recover_empty and set(value) != set(self):
            raise TensorforceError.value(name=name, argument='value', value=value)

        tensor = TensorDict()
        for name, spec in super(NestedDict, self).items():
            if recover_empty and name not in value:
                assert not isinstance(spec, self.value_type) and len(spec) == 0
                tensor[name] = spec.to_tensor(
                    value=None, batched=batched, recover_empty=recover_empty, name=name
                )
            else:
                tensor[name] = spec.to_tensor(
                    value=value[name], batched=batched, recover_empty=recover_empty, name=name
                )
        return tensor

    def from_tensor(self, *, tensor, batched, name='TensorSpec.from_tensor'):
        if not isinstance(tensor, TensorDict):
            raise TensorforceError.type(name=name, argument='tensor', dtype=type(tensor))

        # TODO: improve exception message to include invalid keys
        if set(tensor) != set(self):
            raise TensorforceError.value(name=name, argument='tensor', value=tensor)

        value = ArrayDict()
        for name, spec in super(NestedDict, self).items():
            value[name] = spec.from_tensor(tensor=tensor[name], batched=batched, name=name)
        return value

    def np_assert(self, *, x, message, batched=False):
        if not isinstance(x, dict):
            raise TensorforceError(
                message.format(name='', issue=('type {} != dict'.format(type(x))))
            )

        for name, spec, x in self.zip_items(x):
            if name is None:
                name = ''
            spec.np_assert(x=x, message=(
                None if message is None else message.format(name=name, issue='{issue}')
            ))

    def tf_assert(self, *, x, batch_size=None, include_type_shape=False, message=None):
        if not isinstance(x, TensorDict):
            raise TensorforceError(
                message.format(name='', issue=('type {} != TensorDict'.format(type(x))))
            )

        assertions = list()
        for name, spec, x in self.zip_items(x):
            if name is None:
                name = ''
            assertions.extend(spec.tf_assert(
                x=x, batch_size=batch_size, include_type_shape=include_type_shape,
                message=(None if message is None else message.format(name=name, issue='{issue}'))
            ))

        return assertions

    def unify(self, *, other):
        if set(self) != set(other):
            raise TensorforceError.mismatch(
                name='TensorsSpec.unify', argument='keys', value1=sorted(self), value2=sorted(other)
            )
        return self.fmap(function=(lambda x, y: x.unify(other=y)), zip_values=other)

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
