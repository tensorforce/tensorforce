# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

    # modify dtype mappings

    def __init__(
        self, *,
        buffer_observe=False,
        create_debug_assertions=False,
        create_tf_assertions=True,
        device=None,
        eager_mode=False,
        enable_int_action_masking=True,
        name='agent',
        seed=None,
        tf_log_level=40
    ):
        assert buffer_observe is False or buffer_observe == 'episode' or \
            isinstance(buffer_observe, int) and buffer_observe >= 1
        if buffer_observe is False:
            buffer_observe = 1
        super().__setattr__('buffer_observe', buffer_observe)

        assert isinstance(create_debug_assertions, bool)
        super().__setattr__('create_debug_assertions', create_debug_assertions)

        assert isinstance(create_tf_assertions, bool)
        super().__setattr__('create_tf_assertions', create_tf_assertions)

        assert isinstance(eager_mode, bool)
        super().__setattr__('eager_mode', eager_mode)

        assert isinstance(enable_int_action_masking, bool)
        super().__setattr__('enable_int_action_masking', enable_int_action_masking)

        assert device is None or isinstance(device, str)  # more specific?
        super().__setattr__('device', device)

        assert isinstance(name, str)
        super().__setattr__('name', name)

        assert seed is None or isinstance(seed, int)
        super().__setattr__('seed', seed)

        assert isinstance(tf_log_level, int) and tf_log_level >= 0
        super().__setattr__('tf_log_level', tf_log_level)

    def __setattr__(self, name, value):
        raise NotImplementedError

    def __delattr__(self, name):
        raise NotImplementedError
