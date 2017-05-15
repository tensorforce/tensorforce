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


class Configuration(dict):
    """Configuration class that extends dict and reads configuration files (currently only json)
    """

    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, item, value):
        self[item] = value

    def __add__(self, other):
        result = Configuration(self.items())
        result.update(other)
        return result

    def default(self, default):
        for key in default:
            if key not in self:
                self[key] = default[key]

    def read_json(self, filename):
        """Read configuration from filename.

        Args:
            filename: filename/path for configuration file. currently only json is supported.

        Returns: void

        """
        path = os.path.join(os.getcwd(), filename)

        # don't catch, we let open() and json.loads() raise their own exceptions
        with open(path, 'r') as f:
            self.update(json.loads(f.read()))


def create_config(values, default=None):
    """Create ``Configuration object`` from dict. Use ``default`` dict for default values.

    Args:
        values: dict containing actual values.
        default: dict containing default values.

    Returns: ``Configuration`` object.

    """
    if default:
        if isinstance(default, dict):
            default_data = default
        else:
            raise ValueError("Invalid default config data.")
        config = Configuration(default)
        if values:
            config.update(values)
    else:
        config = Configuration(values)
    return config
