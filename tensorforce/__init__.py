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

import logging
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf


tf.compat.v1.disable_eager_execution()


from tensorforce.exception import TensorforceError
from tensorforce import util
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner


# tf.get_logger().setLevel('WARNING')
# tf.autograph.set_verbosity(3)

__all__ = ['Agent', 'Environment', 'Runner', 'TensorforceError']

__version__ = '0.5.4'

logging.getLogger(__name__).addHandler(logging.NullHandler())
