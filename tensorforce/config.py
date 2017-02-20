# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
Configuration class that extends dict and reads configuration files 
(currently only json)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json


class Config(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, item, value):
        self[item] = value

    def __add__(self, other):
        result = Config(self.items())
        result.update(other)
        return result

    def read_json(self, filename):
        """
        Read configuration from json file

        :param filename: path of configuration file
        """
        path = os.path.join(os.getcwd(), filename)

        # don't catch, we let open() and json.loads() raise their own exceptions
        with open(path, 'r') as f:
            self.update(json.loads(f.read()))


def create_config(values, default=None):
    """
    Create Config object from dict. Use default dict for default values.

    :param values: dict containing actual values
    :param default: dict containing default values or string pointing to default file
    :return: Config object
    """
    if default:
        if isinstance(default, dict):
            default_data = default
        else:
            raise ValueError("Invalid default config data.")
        config = Config(default)
        if values:
            config.update(values)
    else:
        config = Config(values)
    return config
