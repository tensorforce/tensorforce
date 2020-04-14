# Copyright 2018 Tensorforce Team. All Rights Reserved.
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


class TensorforceConfig(object):

    def __init__(self, enable_int_action_masking=True, buffer_observe=1, seed=None):
        super().__setattr__('enable_int_action_masking', enable_int_action_masking)
        assert isinstance(buffer_observe, int) and buffer_observe >= 1
        super().__setattr__('buffer_observe', buffer_observe)
        assert seed is None or isinstance(seed, int)
        super().__setattr__('seed', seed)

    def __setattr__(self, name, value):
        raise NotImplementedError

    def __delattr__(self, name):
        raise NotImplementedError

    # exploration as part of independent
    # variable noise
    # sampling
    # modify dtype mappings
