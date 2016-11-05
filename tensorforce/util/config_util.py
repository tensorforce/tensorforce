# Copyright 2016 reinforce.io. All Rights Reserved.
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

"""
Utility functions concerning configurations
"""

import importlib


def get_function(string, param=None, default=None):
    """
    Get function reference by full module path. Either returns the function reference or calls the function
    if param is not None and returns the result.

    :param string: String containing the full function path
    :param param: None to return function name, kwargs dict to return executed function
    :param default: Default reference to return if str is None or empty
    :return: Function reference, or result from function call
    """
    if not string:
        return default
    module_name, function_name = string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    if isinstance(param, dict):
        return func(**param)
    else:
        return func
