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

from tensorforce import util
import tensorforce.core.distributions


class Distribution(object):

    @classmethod
    def from_tensors(cls, tensors):
        raise NotImplementedError

    def get_tensors(self):
        raise NotImplementedError

    def create_tf_operations(self, x, deterministic):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.distribution)

    def sample(self):
        raise NotImplementedError

    def log_probability(self, action):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def kl_divergence(self, other):
        raise NotImplementedError

    @staticmethod
    def from_config(config, kwargs=None):
        return util.get_object(
            obj=config,
            predefined=tensorforce.core.distributions.distributions,
            kwargs=kwargs
        )
