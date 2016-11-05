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
Base class for value functions, contains general tensorflow utility
that all value functions need.
"""

def ValueFunction(object):

    def __init__(self, tf_config):
        """
        Value functions provide the general interface to TensorFlow functionality,
        e.g.
        :param self:
        :param tf_config:
        :return:
        """
        pass

    def load_model(self, model_location):
        raise NotImplementedError

    def save_model(self, export_location):
        raise NotImplementedError