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
Utility functions concerning state wrappers.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.exceptions.tensorforce_exceptions import TensorForceValueError
from tensorforce.state_wrappers import *


def create_wrapper(wrapper_type, config):
    """
    Create wrapper instance by providing type as a string parameter.

    :param wrapper_type: String parameter containing wrapper type
    :param config: Dict containing configuration
    :return: Wrapper instance
    """
    wrapper_class = wrappers.get(wrapper_type)

    if not wrapper_class:
        raise TensorForceValueError("No such wrapper: {}".format(wrapper_type))

    return wrapper_class(config)


wrappers = {
    'ConcatWrapper': ConcatWrapper,
    'AtariWrapper': AtariWrapper
}
