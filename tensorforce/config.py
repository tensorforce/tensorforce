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


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from copy import deepcopy
import json
import os

from tensorforce import TensorForceError


class Configuration(object):
    """Configuration class that extends dict and reads configuration files (currently only json)
    """

    def __init__(self, **kwargs):
        self._config = dict(kwargs)
        self._accessed = {key: False for key, value in kwargs.items() if not isinstance(value, Configuration)}

    def to_json(self, filename):
        with open(filename, 'w') as fp:
            fp.write(json.dumps(self.as_dict()))

    @staticmethod
    def from_json(filename, absolute_path=False):
        if absolute_path:
            path = filename
        else:
            path = os.path.join(os.getcwd(), filename)

        with open(path, 'r') as fp:
            json_string = fp.read()
        return Configuration.from_json_string(json_string=json_string)

    @staticmethod
    def from_json_string(json_string):
        return Configuration(**json.loads(json_string))

    def __getstate__(self):
        return self._config

    def __setstate__(self, d):
        self._config = d

    def __str__(self):
        return '{' + ', '.join('{}={}'.format(key, value) for key, value in self._config.items()) + '}'

    def __len__(self):
        return len(self._config)

    def __iter__(self):
        for key in self._config:
            yield key

    def items(self):
        for key, value in self._config.items():
            yield key, value

    def keys(self):
        for key in self._config.keys():
            yield key

    def values(self):
        for value in self._config.values():
            yield value

    def __contains__(self, key):
        return key in self._config

    def copy(self):
        return Configuration(**deepcopy(self._config))

    def __getattr__(self, key):
        if key not in self._config:
            raise TensorForceError("Value '{}' is not defined.".format(key))
        if key in self._accessed:
            self._accessed[key] = True
        return self._config[key]

    def __setattr__(self, key, value):
        if key == '_config' or key == '_accessed':
            super(Configuration, self).__setattr__(key, value)
        elif key not in self._config:
            raise TensorForceError("Value '{}' is not defined.".format(key))
        else:
            raise TensorForceError("Setting config attributes not allowed.")
            # self._config[key] = value

    def set(self, key, value):
        self._config[key] = value
        self._accessed[key] = False

    def obligatory(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._config:
                raise TensorForceError("Value '{}' should not be defined externally.".format(key))
            self._config[key] = value
            self._accessed[key] = False

    def as_dict(self):
        d = dict()
        for key, value in self._config.items():
            if isinstance(value, Configuration):
                d[key] = value.as_dict()
            else:
                d[key] = value
        return d

    def default(self, default):
        for key, value in default.items():
            if key not in self._config:
                self._config[key] = value
                self._accessed[key] = False

    def not_accessed(self):
        not_accessed = list()
        for key, value in self._config.items():
            if isinstance(value, Configuration):
                for subkey in value.not_accessed():
                    not_accessed.append('{}.{}'.format(key, subkey))
            elif not self._accessed[key]:
                not_accessed.append(key)
        return not_accessed
