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

import importlib
import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError


epsilon = 1e-6


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def shape(x, unknown=-1):
    return tuple(unknown if dims is None else dims for dims in x.get_shape().as_list())


def cumulative_discount(rewards, terminals, discount):
    if discount == 0.0:
        return rewards
    cumulative = 0.0
    for n, (reward, terminal) in reversed(list(enumerate(zip(rewards, terminals)))):
        if terminal:
            cumulative = 0.0
        cumulative = reward + cumulative * discount
        rewards[n] = cumulative
    return rewards


def np_dtype(dtype):
    if dtype == 'float' or dtype == float:
        return np.float32
    elif dtype == 'int' or dtype == int:
        return np.int32
    elif dtype == 'bool' or dtype == bool:
        return np.bool_
    else:
        raise TensorForceError()


def tf_dtype(dtype):
    if dtype == 'float' or dtype == float:
        return tf.float32
    elif dtype == 'int' or dtype == int:
        return tf.int32
    else:
        raise TensorForceError()


def function(f, predefined=None):
    if predefined is not None and f in predefined:
        return predefined[f]
    module_name, function_name = f.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)



# def make_function(data, fk):
#     """
#     Take data dict and convert string function reference with key `fk` to a real function reference, using
#     `fk`_args as *args and `fk`_kwargs as **kwargs, removing these keys from the data dict.

#     :param data: data dict
#     :param fk: string function key
#     :return: boolean
#     """
#     fn = data.get(fk)

#     if fn is None:
#         return True
#     elif callable(fn):
#         return True
#     else:
#         args_val = "{}_args".format(fk)
#         kwargs_val = "{}_kwargs".format(fk)

#         args = data.pop(args_val, None)
#         kwargs = data.pop(kwargs_val, None)

#         func = get_function(fn)

#         if args is None and kwargs is None:
#             # If there are no args and no kwargs, just return the function reference
#             data[fk] = func
#             return True

#         # Otherwise, call the function
#         if args is None:
#             args = []
#         if kwargs is None:
#             kwargs = {}

#         data[fk] = func(*args, **kwargs)
#         return True




# def repeat_action(environment, action, repeat_action=1):
#     """
#     Repeat action `repeat_action_count` times. Cumulate reward and return last state.
#
#     :param environment: Environment object
#     :param action: Action to be executed
#     :param repeat_action: How often to repeat the action
#     :return: result dict
#     """
#     if repeat_action <= 0:
#         raise ValueError('repeat_action lower or equal zero')
#
#     reward = 0.
#     terminal_state = False
#     for count in xrange(repeat_action):
#         result = environment.execute_action(action)
#
#         state = result['state']
#         reward += result['reward']
#         terminal_state = terminal_state or result['terminal_state']
#         info = result.get('info', None)
#
#     return dict(state=state,
#                 reward=reward,
#                 terminal_state=terminal_state,
#                 info=info)
#







# preprocessors = {
#     'concat': preprocessing.Concat,
#     'grayscale': preprocessing.Grayscale,
#     'imresize': preprocessing.Imresize,
#     'maximum': preprocessing.Maximum,
#     'normalize': preprocessing.Normalize,
#     'standardize': preprocessing.Standardize
# }
#
#
# def build_preprocessing_stack(config):
#     stack = preprocessing.Stack()
#
#     for preprocessor_conf in config:
#         preprocessor_name = preprocessor_conf[0]
#
#         preprocessor_params = []
#         if len(preprocessor_conf) > 1:
#             preprocessor_params = preprocessor_conf[1:]
#
#         preprocessor_class = preprocessors.get(preprocessor_name, None)
#         if not preprocessor_class:
#             raise ConfigError("No such preprocessor: {}".format(preprocessor_name))
#
#         preprocessor = preprocessor_class(*preprocessor_params)
#         stack += preprocessor
#
#     return stack



# def create_agent(agent_type, config, scope='prefixed_scope'):
#     """
#     Create agent instance by providing type as a string parameter.
#
#     :param agent_type: String parameter containing agent type
#     :param config: Dict containing configuration
#     :param scope: Scope prefix used for distributed tensorflow scope separation
#     :return: Agent instance
#     """
#     agent_class = agents.get(agent_type)
#
#     if not agent_class:
#         raise TensorForceError("No such agent: {}".format(agent_type))
#
#     return agent_class(config, scope)


# def get_default_config(agent_type):
#     """
#     Get default configuration from agent by providing type as a string parameter.
#
#     :param agent_type: String parameter containing agent type
#     :return: Default configuration dict
#     """
#     agent_class = agents.get(agent_type)
#
#     if not agent_class:
#         raise TensorForceError("No such agent: {}".format(agent_type))
#
#     return Configuration(agent_class.default_config), Config(agent_class.model_ref.default_config)


#
# agents = {
#     'RandomAgent': RandomAgent,
#     'DQNAgent': DQNAgent,
#     'NAFAgent': NAFAgent,
#     'TRPOAgent': TRPOAgent,
#     'VPGAgent': VPGAgent,
#     'DQFDAgent': DQFDAgent,
# }
