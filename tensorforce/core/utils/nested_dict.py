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

    _SINGLETON = 'SINGLETON'

    def __init__(self, arg=None, *, value_type=None, overwrite=None, singleton=None, **kwargs):
        super().__init__()
        super().__setattr__('value_type', value_type)
        super().__setattr__('overwrite', overwrite)
        if singleton is not None:
            if arg is not None or len(kwargs) > 0:
                raise TensorforceError.invalid(name='NestedDict', argument='singleton')
            self[None] = singleton
        elif arg is None:
            self.update(**kwargs)
        else:
            self.update(arg, **kwargs)

    def copy(self):
        if self.is_singleton():
            x = self.__class__()
            value = self.singleton()
            x[None] = value.copy() if hasattr(value, 'copy') else value
        else:
            x = self.__class__((
                (name, (value.copy() if hasattr(value, 'copy') else value))
                for name, value in super().items()
            ))
        super(NestedDict, x).__setattr__('value_type', self.value_type)
        super(NestedDict, x).__setattr__('overwrite', self.overwrite)
        return x

    def flatten(self):
        return list(self.values())

    def zip_items(self, *others):
        assert all(len(other) == len(self) for other in others)
        for name, value in self.items():
            assert all(name in other for other in others)
            other_values = tuple(other[name] for other in others)
            yield (name, value) + other_values

    def fmap(self, *, function, cls=None, with_names=False, zip_values=None):
        if cls is None:
            # Use same class and settings for mapped dict
            values = self.__class__()
            super(NestedDict, values).__setattr__('value_type', self.value_type)
            super(NestedDict, values).__setattr__('overwrite', self.overwrite)
            setitem = values.__setitem__
        elif issubclass(cls, list):
            # Special target class list implies flatten
            values = cls()
            setitem = (lambda n, v: (values.extend(v) if isinstance(v, cls) else values.append(v)))
        elif issubclass(cls, dict):
            # Custom target class
            values = cls()
            setitem = values.__setitem__
        else:
            raise TensorforceError.value(name='NestedDict.fmap', argument='cls', value=cls)

        for name, value in super().items():
            if name == self.__class__._SINGLETON:
                name = None

            if isinstance(with_names, str):
                if name is None:
                    full_name = with_names
                else:
                    full_name = '{}/{}'.format(with_names, name)
            else:
                assert isinstance(with_names, bool)
                if with_names:
                    if name is None and not isinstance(value, self.value_type):
                        full_name = True
                    else:
                        full_name = name
                else:
                    full_name = False

            if isinstance(zip_values, (tuple, list)):
                zip_value = tuple(xs[name] for xs in zip_values)
            elif isinstance(zip_values, NestedDict):
                zip_value = (zip_values[name],)
            elif zip_values is None:
                zip_value = None
            else:
                raise TensorforceError.type(
                    name='NestedDict.fmap', argument='zip_values', dtype=type(zip_values)
                )

            if isinstance(value, self.value_type):
                if with_names:
                    args = (full_name, value)
                else:
                    args = (value,)
                if zip_value is not None:
                    args += zip_value
                    # args += tuple(
                    #     x.singleton() if isinstance(x, NestedDict) else x for x in zip_value
                    # )
                setitem(name, function(*args))
            else:
                setitem(name, value.fmap(
                    function=function, cls=cls, with_names=full_name, zip_values=zip_value
                ))

        return values

    def is_singleton(self):
        return super().__len__() == 1 and super().__contains__(self.__class__._SINGLETON)

    def singleton(self):
        assert self.is_singleton()
        return super().__getitem__(self.__class__._SINGLETON)

    def __len__(self):
        return sum(
            1 if isinstance(value, self.value_type) else len(value) for value in super().values()
        )

    def __iter__(self):
        for name, value in super().items():
            if name == self.__class__._SINGLETON:
                if isinstance(value, self.value_type):
                    yield None
                else:
                    yield from value
            elif isinstance(value, self.value_type):
                yield name
            else:
                assert isinstance(value, self.__class__)
                for subname in value:
                    if subname is None:
                        yield name
                    else:
                        yield '{}/{}'.format(name, subname)

    def items(self):
        for name, value in super().items():
            if name == self.__class__._SINGLETON:
                if isinstance(value, self.value_type):
                    yield None, value
                else:
                    yield from value.items()
            elif isinstance(value, self.value_type):
                yield name, value
            else:
                assert isinstance(value, self.__class__)
                for subname, subvalue in value.items():
                    if subname is None:
                        yield name, subvalue
                    else:
                        yield '{}/{}'.format(name, subname), subvalue

    def values(self):
        for value in super().values():
            if isinstance(value, self.value_type):
                yield value
            else:
                assert isinstance(value, self.__class__)
                yield from value.values()

    def __contains__(self, item):
        if item is None or item == self.__class__._SINGLETON:
            assert super().__len__() == 0 or self.is_singleton()
            return super().__contains__(self.__class__._SINGLETON)

        elif isinstance(item, (list, tuple)):
            for name in item:
                if name not in self:
                    return False
            return True

        elif not isinstance(item, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(item))

        elif item.startswith(self.__class__._SINGLETON + '/'):
            raise TensorforceError.value(name='NestedDict', argument='item', value=item)

        elif self.is_singleton():
            value = self.singleton()
            if isinstance(value, self.value_type):
                return False
            else:
                return item in value

        elif '/' in item:
            item, subitem = item.split('/', 1)
            if super().__contains__(item):
                value = super().__getitem__(item)
                assert isinstance(value, self.__class__)
                return subitem in value
            else:
                return False

        else:
            return super().__contains__(item)

    def __getitem__(self, key):
        if key is None or key == self.__class__._SINGLETON:
            assert self.is_singleton()
            return super().__getitem__(self.__class__._SINGLETON)

        elif isinstance(key, (int, slice)):
            return self.fmap(function=(lambda x: x[key]))

        elif isinstance(key, (list, tuple)):
            return self.__class__(((name, self[name]) for name in key))

        elif not isinstance(key, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(key))

        elif key.startswith(self.__class__._SINGLETON + '/'):
            raise TensorforceError.value(name='NestedDict', argument='key', value=key)

        elif self.is_singleton():
            return self.singleton()[key]

        elif '/' in key:
            key, subkey = key.split('/', 1)
            value = super().__getitem__(key)
            assert isinstance(value, self.__class__)
            return value[subkey]

        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, self.value_type):
            if isinstance(value, self.__class__):
                value = value.copy()
            else:
                value = self.__class__(value)
        if not isinstance(value, self.__class__) and not isinstance(value, self.value_type):
            raise TensorforceError.type(name='NestedDict', argument='value', dtype=type(value))

        if key is None or key == self.__class__._SINGLETON:
            assert super().__len__() == 0 or self.is_singleton()
            super().__setitem__(self.__class__._SINGLETON, value)

        elif not isinstance(key, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(key))

        elif key.startswith(self.__class__._SINGLETON + '/'):
            raise TensorforceError.value(name='NestedDict', argument='key', value=key)

        elif self.is_singleton():
            self.singleton()[key] = value

        elif '/' in key:
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
            assert isinstance(value, self.__class__)
            value[subkey] = subvalue
            if not super().__contains__(key):
                # After setting subkey since setitem may modify value (TrackableNestedDict)
                self[key] = value

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

    def get(self, key, default=None):
        if isinstance(key, (list, tuple)):
            return tuple(self.get(key=x, default=default) for x in key)
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
        if key is None or key == self.__class__._SINGLETON:
            assert super().__len__() == 0 or self.is_singleton()
            if super().__contains__(self.__class__._SINGLETON):
                value = super().__getitem__(self.__class__._SINGLETON)
                super().__delitem__(self.__class__._SINGLETON)
            else:
                value = default
            return value

        elif not isinstance(key, str):
            raise TensorforceError.type(name='NestedDict', argument='key', dtype=type(key))

        elif key.startswith(self.__class__._SINGLETON + '/'):
            raise TensorforceError.value(name='NestedDict', argument='key', value=key)

        elif self.is_singleton():
            value = self.singleton()
            if isinstance(value, self.value_type):
                return default
            else:
                return value.pop(key, default=default)

        elif '/' in key:
            key, subkey = key.split('/', 1)
            value = super().__getitem__(key)
            assert isinstance(value, self.__class__)
            return value.pop(subkey, default)

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
