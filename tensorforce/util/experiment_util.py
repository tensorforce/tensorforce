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

import numpy as np

from six.moves import xrange

from tensorforce.exceptions import ConfigError
from tensorforce import preprocessing

preprocessors = {
    'concat': preprocessing.Concat,
    'grayscale': preprocessing.Grayscale,
    'imresize': preprocessing.Imresize,
    'maximum': preprocessing.Maximum,
    'normalize': preprocessing.Normalize,
    'standardize': preprocessing.Standardize
}

def global_seed():
    """
    Convenience function to control random seeding throughout the framework.
    :return: A numpy random number generator with a fixed seed.
    """
    return np.random.RandomState(42)


def get_path(continuous, episode):
    """
    Finalises an episode and turns it into a dict pointing to numpy arrays.
    :return:
    """
    path = {'states': np.concatenate(np.expand_dims(episode['states'], 0)),
            'actions': np.array(episode['actions']),
            'terminated': episode['terminated'],
            'action_means': np.array(episode['action_means']),
            'rewards': np.array(episode['rewards'])}

    if continuous:
        path['action_log_stds'] = np.concatenate(episode['action_log_stds'])

    return path

def repeat_action(environment, action, repeat_action=1):
    """
    Repeat action `repeat_action_count` times. Cumulate reward and return last state.

    :param environment: Environment object
    :param action: Action to be executed
    :param repeat_action_count: How often to repeat the action
    :return: result dict
    """
    if repeat_action <= 0:
        raise ValueError('repeat_action lower or equal zero')

    reward = 0.
    terminal_state = False
    for count in xrange(repeat_action):
        result = environment.execute_action(action)

        state = result['state']
        reward += result['reward']
        terminal_state = terminal_state or result['terminal_state']
        info = result.get('info', None)

    return dict(state=state,
                reward=reward,
                terminal_state=terminal_state,
                info=info)

def build_preprocessing_stack(config):
    stack = preprocessing.Stack()

    for preprocessor_conf in config:
        preprocessor_name = preprocessor_conf[0]

        preprocessor_params = []
        if len(preprocessor_conf) > 1:
            preprocessor_params = preprocessor_conf[1:]

        preprocessor_class = preprocessors.get(preprocessor_name, None)
        if not preprocessor_class:
            raise ConfigError("No such preprocessor: {}".format(preprocessor_name))

        preprocessor = preprocessor_class(*preprocessor_params)
        stack += preprocessor

    return stack
