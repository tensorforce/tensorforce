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

from tensorforce import TensorforceError


def _is_keyword(x):
    return x in {'bool', 'int', 'float', 'type', 'shape', 'min_value', 'max_value', 'num_values'}


class NestedDict(OrderedDict):

    def __init__(self, *args, value_type=object, overwrite=False, **kwargs):
        super().__init__()
        super().__setattr__('value_type', value_type)
        super().__setattr__('overwrite', overwrite)
        if len(args) > 1:
            raise TensorforceError.invalid(
                name='NestedDict', argument='*args', condition='more than one'
            )
        self.update(*args, **kwargs)

    def copy(self):
        x = self.__class__((
            (name, (value.copy() if hasattr(value, 'copy') else value))
            for name, value in super().items()
        ))
        super(NestedDict, x).__setattr__('value_type', self.value_type)
        super(NestedDict, x).__setattr__('overwrite', self.overwrite)
        return x

    def fmap(self, function, cls=None, with_names=False):
        if cls is None:
            x = self.__class__()
            super(NestedDict, x).__setattr__('value_type', self.value_type)
            super(NestedDict, x).__setattr__('overwrite', self.overwrite)
        else:
            x = cls()
        for name, value in self.items():
            if with_names:
                x[name] = function(name, value)
            else:
                x[name] = function(value)
        return x

    def __len__(self):
        return sum(
            len(value) if isinstance(value, self.__class__) else 1 for value in super().values()
        )

    def __iter__(self):
        for name, value in super().items():
            if isinstance(value, self.__class__):
                for subname in value:
                    yield '{}/{}'.format(name, subname)
            elif isinstance(value, self.value_type):
                yield name
            else:
                raise TensorforceError.unexpected()

    def items(self):
        for name, value in super().items():
            if isinstance(value, self.__class__):
                for subname, subvalue in value.items():
                    yield '{}/{}'.format(name, subname), subvalue
            elif isinstance(value, self.value_type):
                yield name, value
            else:
                raise TensorforceError.unexpected()

    def values(self):
        for value in super().values():
            if isinstance(value, self.__class__):
                yield from value.values()
            elif isinstance(value, self.value_type):
                yield value
            else:
                raise TensorforceError.unexpected()

    def __contains__(self, item):
        if isinstance(item, tuple):
            for name in item:
                if name not in self:
                    return False
            return True

        elif not isinstance(item, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(item))

        elif '/' in item:
            item, subitem = item.split('/', 1)
            if super().__contains__(item):
                value = super().__getitem__(item)
                if isinstance(value, self.__class__):
                    return subitem in value
                else:
                    raise TensorforceError.unexpected()
            else:
                return False

        else:
            return super().__contains__(item)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.__class__(((name, value) for name, value in super().items() if name in key))

        elif not isinstance(key, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(key))

        elif '/' in key:
            key, subkey = key.split('/', 1)
            value = super().__getitem__(key)
            if isinstance(value, self.__class__):
                return value[subkey]
            else:
                raise TensorforceError.unexpected()

        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(key))

        if isinstance(value, dict) and not isinstance(value, self.value_type):
            if isinstance(value, self.__class__):
                value = value.copy()
            else:
                value = self.__class__(value)
        if not isinstance(value, self.__class__) and not isinstance(value, self.value_type):
            raise TensorforceError.type(name='NestedDict', argument='value', dtype=type(value))

        if '/' in key:
            subvalue = value
            key, subkey = key.split('/', 1)
            if _is_keyword(x=key):
                raise TensorforceError.value(
                    name='NestedDict', argument='key', value=key, hint='reserved keyword'
                )
            if super().__contains__(key):
                value = super().__getitem__(key)
            else:
                value = self.__class__()
                super(NestedDict, value).__setattr__('value_type', self.value_type)
                super(NestedDict, value).__setattr__('overwrite', self.overwrite)
                super().__setitem__(key, value)
            if isinstance(value, self.__class__):
                value[subkey] = subvalue
            else:
                raise TensorforceError.unexpected()

        else:
            if _is_keyword(x=key):
                raise TensorforceError.value(
                    name='NestedDict', argument='key', value=key, hint='reserved keyword'
                )
            if not self.overwrite and super().__contains__(key):
                raise TensorforceError.value(
                    name='NestedDict', argument='key', value=key, condition='already set'
                )
            super().__setitem__(key, value)

    def __repr__(self):
        return '{type}({items})'.format(type=self.__class__.__name__, items=', '.join(
            '{key}={value}'.format(key=key, value=value) for key, value in super().items()
        ))

    def item(self):
        return next(iter(self.values()))

    def get(self, key, *args, default=None):
        if len(args) > 0:
            return tuple(self.get(key=x, default=default) for x in (key,) + args)
        elif key in self:
            return self[key]
        else:
            return default

    def update(self, other=None, **kwargs):
        if other is not None:
            if hasattr(other, 'items'):
                other = other.items()
            for key, value in other:
                if key in kwargs:
                    raise TensorforceError.value(
                        name='NestedDict.update', argument='key', value=key,
                        condition='specified twice'
                    )
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    __str__ = __repr__

    has_key = __contains__

    keys = __iter__
    iterkeys = __iter__
    viewkeys = __iter__

    itervalues = values
    viewvalues = values

    iteritems = items
    viewitems = items

    def __setattr__(self, name, value):
        raise NotImplementedError

    def __delattr__(self, name):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    @classmethod
    def fromkeys(cls, iterable, value=None):
        raise NotImplementedError

    def pop(self, key, default=None):
        raise NotImplementedError

    def popitem(self):
        raise NotImplementedError

    def setdefault(self, key, default=None):
        raise NotImplementedError
