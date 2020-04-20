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

    def __init__(self, arg=None, *, value_type, overwrite, **kwargs):
        super().__init__()
        super().__setattr__('value_type', value_type)
        super().__setattr__('overwrite', overwrite)
        if arg is None:
            self.update(**kwargs)
        else:
            self.update(arg, **kwargs)

    def copy(self):
        x = self.__class__((
            (name, (value.copy() if hasattr(value, 'copy') else value))
            for name, value in super().items()
        ))
        super(NestedDict, x).__setattr__('value_type', self.value_type)
        super(NestedDict, x).__setattr__('overwrite', self.overwrite)
        return x

    def fmap(self, function, cls=None, with_names=False, zip_values=None):
        if cls is None:
            # Use same class and settings for mapped dict
            values = self.__class__()
            super(NestedDict, values).__setattr__('value_type', self.value_type)
            super(NestedDict, values).__setattr__('overwrite', self.overwrite)
            setitem = values.__setitem__

        elif cls is list:
            # Special target class list implies flatten
            values = list()
            setitem = (lambda n, v: (values.extend(v) if isinstance(v, list) else values.append(v)))

        else:
            # Custom target class
            assert issubclass(cls, NestedDict)
            values = cls()
            setitem = values.__setitem__

        for name, value in super().items():
            # with_names
            if isinstance(with_names, str):
                full_name = '{}/{}'.format(with_names, name)
            elif with_names is True:
                full_name = name
            else:
                full_name = False

            # zip_values
            if zip_values is None:
                zip_value = None
            elif isinstance(zip_values, NestedDict):
                zip_value = (zip_values[name],)
            elif isinstance(zip_values, tuple):
                zip_value = tuple(xs[name] for xs in zip_values)
            else:
                raise TensorforceError.type(
                    name='NestedDict.fmap', argument='zip_values', dtype=type(zip_values)
                )

            # Recursive fmap call
            if isinstance(value, self.__class__):
                setitem(name, value.fmap(
                    function=function, cls=cls, with_names=full_name, zip_values=zip_value
                ))

            # Actual value mapping
            else:
                if full_name is False:
                    if zip_value is None:
                        setitem(name, function(value))
                    else:
                        setitem(name, function(value, *zip_value))
                else:
                    if zip_value is None:
                        setitem(name, function(full_name, value))
                    else:
                        setitem(name, function(full_name, value, *zip_value))

        return values

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
            return self.__class__(((name, self[name]) for name in key))

        elif not isinstance(key, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(key))

        elif '/' in key:
            key, subkey = key.split('/', 1)
            value = super().__getitem__(key)
            if isinstance(value, self.__class__):
                return value[subkey]
            else:
                raise TensorforceError.unexpected()

        # else:
        #     return super().__getitem__(key)

        elif super().__contains__(key):
            return super().__getitem__(key)

        else:
            value = self.__class__()
            super(NestedDict, value).__setattr__('value_type', self.value_type)
            super(NestedDict, value).__setattr__('overwrite', self.overwrite)
            return value

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

    def key(self):
        return next(iter(self))

    def value(self):
        return next(iter(self.values()))

    def item(self):
        return next(iter(self.items()))

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

    def pop(self, key, default=None):
        if not isinstance(key, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(key))

        elif '/' in key:
            key, subkey = key.split('/', 1)
            value = super().__getitem__(key)
            if isinstance(value, self.__class__):
                return value.pop(subkey, default)
            else:
                raise TensorforceError.unexpected()

        else:
            # TODO: can't use pop since __delitem__ not implemented
            if super().__contains__(key):
                value = super().__getitem__(key)
                super().__delitem__(key)
            else:
                value = default
            return value

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

    def popitem(self):
        raise NotImplementedError

    def setdefault(self, key, default=None):
        raise NotImplementedError
