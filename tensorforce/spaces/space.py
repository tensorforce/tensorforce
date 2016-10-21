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
Base space implementation
"""

class Space(object):

    def contains(self, x):
        """
        Returns true if space contains item x, otherwise false

        :param x: item to be checked
        :return: boolean
        """
        raise NotImplementedError

    def flatten(self, x):
        """
        Flatten item x

        :param x: unflattened item
        :return: flattened item
        """
        raise NotImplementedError

    def unflatten(self, x):
        """
        Unflatten item x

        :param x: flattened item
        :return: unflattened item
        """
        raise NotImplementedError

