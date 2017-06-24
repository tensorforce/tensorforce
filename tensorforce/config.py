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

import os
import json

from tensorforce import TensorForceError


class Configuration(object):
    """Configuration class that extends dict and reads configuration files (currently only json)
    """

    def __init__(self, **kwargs):
        self._config = dict(**kwargs)

    @staticmethod
    def from_json(filename):
        path = os.path.join(os.getcwd(), filename)
        with open(path, 'r') as fp:
            config = json.load(fp=fp)
        return Configuration(**config)

    def __str__(self):
        return '{' + ', '.join('{}={}'.format(key, value) for key, value in self._config.items()) + '}'

    def __iter__(self):
        return iter(self._config.items())

    def items(self):
        return self._config.items()

    def __contains__(self, name):
        return name in self._config

    def __getattr__(self, name):
        if name not in self._config:
            raise TensorForceError('Value for `{}` is not defined.'.format(name))
        return self._config[name]

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setattr__(self, name, value):
        if name == '_config':
            value = {k: make_config_value(v) for k, v in value.items()}
            super(Configuration, self).__setattr__(name, value)
        elif name not in self._config:
            raise TensorForceError('Value is not defined.')
        else:
            self._config[name] = make_config_value(value)

    def keys(self):
        return self._config.keys()

    def copy(self):
        return Configuration(**self._config)

    def default(self, default):
        for key, value in default.items():
            if key not in self._config:
                if isinstance(value, dict):
                    value = Configuration(**value)
                self._config[key] = value


def make_config_value(value):
    if isinstance(value, dict):
        return Configuration(**value)
    elif isinstance(value, list):
        return [make_config_value(v) for v in value]
    else:
        return value
