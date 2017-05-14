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


class Distribution(object):

    epsilon = 1e-6

    @classmethod
    def kl_divergence(cls, distr_a, distr_b):
        """
        Get KL divergence between distributions a and b.
        """
        raise NotImplementedError

    @classmethod
    def entropy(cls, distribution):
        """
        Get current entropy, mainly used for debugging purposes.
        """
        raise NotImplementedError

    def create_tf_operations(self, x, sample=True):
        raise NotImplementedError

    def log_probability(self, action):
        raise NotImplementedError
