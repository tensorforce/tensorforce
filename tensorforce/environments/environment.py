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

import importlib
import json
import math
import os
import random
import sys
from threading import Thread
import time
from traceback import format_tb

import numpy as np

from tensorforce import TensorforceError, util
from tensorforce.core import TensorSpec, TensorsSpec
import tensorforce.environments


class Environment(object):
    """
    Tensorforce environment interface.
    """

    @staticmethod
    def create(
        environment=None, max_episode_timesteps=None, reward_shaping=None,
        remote=None, blocking=False, host=None, port=None, **kwargs
    ):
        """
        Creates an environment from a specification. In case of "socket-server" remote mode, runs
        environment in server communication loop until closed.

        Args:
            environment (specification | Environment class/object): JSON file, specification key,
                configuration dictionary, library module, `Environment` class/object, or gym.Env
                (<span style="color:#C00000"><b>required</b></span>, invalid for "socket-client"
                remote mode).
            max_episode_timesteps (int > 0): Maximum number of timesteps per episode, overwrites
                the environment default if defined
                (<span style="color:#00C000"><b>default</b></span>: environment default, invalid
                for "socket-client" remote mode).
            reward_shaping (callable[(s,a,t,r,s') -> r|(r,t)] | str): Reward shaping function
                mapping state, action, terminal, reward and next state to shaped reward and
                terminal, or a string expression with arguments "states", "actions", "terminal",
                "reward" and "next_states", e.g. "-1.0 if terminal else max(reward, 0.0)"
                (<span style="color:#00C000"><b>default</b></span>: no reward shaping).
            remote ("multiprocessing" | "socket-client" | "socket-server"): Communication mode for
                remote environment execution of parallelized environment execution, "socket-client"
                mode requires a corresponding "socket-server" running, and "socket-server" mode
                runs environment in server communication loop until closed
                (<span style="color:#00C000"><b>default</b></span>: local execution).
            blocking (bool): Whether remote environment calls should be blocking
                (<span style="color:#00C000"><b>default</b></span>: not blocking, invalid unless
                "multiprocessing" or "socket-client" remote mode).
            host (str): Socket server hostname or IP address
                (<span style="color:#C00000"><b>required</b></span> only for "socket-client" remote
                mode).
            port (int): Socket server port
                (<span style="color:#C00000"><b>required</b></span> only for "socket-client/server"
                remote mode).
            kwargs: Additional arguments.
        """
        if remote not in ('multiprocessing', 'socket-client'):
            if blocking:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='blocking',
                    condition='no multiprocessing/socket-client instance'
                )
        if remote not in ('socket-client', 'socket-server'):
            if host is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='host', condition='no socket instance'
                )
            elif port is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='port', condition='no socket instance'
                )

        if remote == 'multiprocessing':
            from tensorforce.environments import MultiprocessingEnvironment
            environment = MultiprocessingEnvironment(
                blocking=blocking, environment=environment,
                max_episode_timesteps=max_episode_timesteps, reward_shaping=reward_shaping, **kwargs
            )
            return environment

        elif remote == 'socket-client':
            if environment is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='environment',
                    condition='socket-client instance'
                )
            elif max_episode_timesteps is not None:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='max_episode_timesteps',
                    condition='socket-client instance'
                )
            elif len(kwargs) > 0:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='kwargs',
                    condition='socket-client instance'
                )
            from tensorforce.environments import SocketEnvironment
            environment = SocketEnvironment(host=host, port=port, blocking=blocking)
            return environment

        elif remote == 'socket-server':
            from tensorforce.environments import SocketEnvironment
            SocketEnvironment.remote(
                port=port, environment=environment, max_episode_timesteps=max_episode_timesteps,
                reward_shaping=reward_shaping, **kwargs
            )

        elif remote is not None:
            raise TensorforceError.value(name='Environment.create', argument='remote', value=remote)

        elif isinstance(environment, (EnvironmentWrapper, RemoteEnvironment)):
            if max_episode_timesteps is not None and \
                    max_episode_timesteps != environment.max_episode_timesteps():
                raise TensorforceError(
                    message='Environment argument max_episode_timesteps has been specified twice '
                    'with different values: {} != {}.'.format(
                        max_episode_timesteps, environment.max_episode_timesteps()
                    )
                )
            if len(kwargs) > 0:
                raise TensorforceError.invalid(
                    name='Environment.create', argument='kwargs',
                    condition='EnvironmentWrapper instance'
                )
            return environment

        elif isinstance(environment, type) and \
                issubclass(environment, (EnvironmentWrapper, RemoteEnvironment)):
            raise TensorforceError.type(
                name='Environment.create', argument='environment', dtype=type(environment)
            )

        elif isinstance(environment, Environment):
            return EnvironmentWrapper(
                environment=environment, max_episode_timesteps=max_episode_timesteps,
                reward_shaping=reward_shaping
            )

        elif isinstance(environment, type) and issubclass(environment, Environment):
            environment = environment(**kwargs)
            assert isinstance(environment, Environment)
            return Environment.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps,
                reward_shaping=reward_shaping
            )

        elif isinstance(environment, dict):
            # Dictionary specification
            util.deep_disjoint_update(target=kwargs, source=environment)
            environment = kwargs.pop('environment', kwargs.pop('type', 'default'))
            assert environment is not None
            if max_episode_timesteps is None:
                max_episode_timesteps = kwargs.pop('max_episode_timesteps', None)
            if reward_shaping is None:
                reward_shaping = kwargs.pop('reward_shaping', None)

            return Environment.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps,
                reward_shaping=reward_shaping, **kwargs
            )

        elif isinstance(environment, str):
            if os.path.isfile(environment):
                # JSON file specification
                with open(environment, 'r') as fp:
                    environment = json.load(fp=fp)

                util.deep_disjoint_update(target=kwargs, source=environment)
                environment = kwargs.pop('environment', kwargs.pop('type', 'default'))
                assert environment is not None
                if max_episode_timesteps is None:
                    max_episode_timesteps = kwargs.pop('max_episode_timesteps', None)
                if reward_shaping is None:
                    reward_shaping = kwargs.pop('reward_shaping', None)

                return Environment.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps,
                    reward_shaping=reward_shaping, **kwargs
                )

            elif '.' in environment:
                # Library specification
                library_name, module_name = environment.rsplit('.', 1)
                library = importlib.import_module(name=library_name)
                environment = getattr(library, module_name)
                return Environment.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps,
                    reward_shaping=reward_shaping, **kwargs
                )

            elif environment in tensorforce.environments.environments:
                # Keyword specification
                environment = tensorforce.environments.environments[environment]
                return Environment.create(
                    environment=environment, max_episode_timesteps=max_episode_timesteps,
                    reward_shaping=reward_shaping, **kwargs
                )

            else:
                # Default: OpenAI Gym
                try:
                    return Environment.create(
                        environment='gym', level=environment,
                        max_episode_timesteps=max_episode_timesteps, reward_shaping=reward_shaping,
                        **kwargs
                    )
                except TensorforceError:
                    raise TensorforceError.value(
                        name='Environment.create', argument='environment', value=environment
                    )

        else:
            # Default: OpenAI Gym
            from gym import Env
            if isinstance(environment, Env) or \
                    (isinstance(environment, type) and issubclass(environment, Env)):
                return Environment.create(
                    environment='gym', level=environment,
                    max_episode_timesteps=max_episode_timesteps, reward_shaping=reward_shaping,
                    **kwargs
                )

            else:
                raise TensorforceError.type(
                    name='Environment.create', argument='environment', dtype=type(environment)
                )

    def __init__(self):
        # first two arguments, if applicable: level, visualize=False
        util.overwrite_staticmethod(obj=self, function='create')
        self._expect_receive = None
        self._actions = None
        self._num_parallel = None
        self._reset_output_check = True
        self._execute_output_check = True

    def __str__(self):
        return self.__class__.__name__

    def states(self):
        """
        Returns the state space specification.

        Returns:
            specification: Arbitrarily nested dictionary of state descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; state data type
            (<span style="color:#00C000"><b>default</b></span>: "float").</li>
            <li><b>shape</b> (<i>int | iter[int]</i>) &ndash; state shape
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>num_states</b> (<i>int > 0</i>) &ndash; number of discrete state values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum state value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        raise NotImplementedError

    def actions(self):
        """
        Returns the action space specification.

        Returns:
            specification: Arbitrarily nested dictionary of action descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; action data type
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>shape</b> (<i>int > 0 | iter[int > 0]</i>) &ndash; action shape
            (<span style="color:#00C000"><b>default</b></span>: scalar).</li>
            <li><b>num_actions</b> (<i>int > 0</i>) &ndash; number of discrete action values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum action value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        raise NotImplementedError

    def max_episode_timesteps(self):
        """
        Returns the maximum number of timesteps per episode.

        Returns:
            int: Maximum number of timesteps per episode.
        """
        return None

    def is_vectorizable(self):
        """
        Returns true if the environment is vectorizable.

        Returns:
            bool: True if the environment is vectorizable.
        """
        return False

    def close(self):
        """
        Closes the environment.
        """
        pass

    def reset(self, num_parallel=None):
        """
        Resets the environment to start a new episode.

        Args:
            num_parallel (int >= 1): Number of environment instances executed in parallel, only
                valid if environment is vectorizable
                (<span style="color:#00C000"><b>no vectorization</b></span>).

        Returns:
            (parallel,) dict[state]: Dictionary containing initial state(s) and auxiliary
            information, and parallel index vector in case of vectorized execution.
        """
        raise NotImplementedError

    def execute(self, actions):
        """
        Executes the given action(s) and advances the environment by one step.

        Args:
            actions (dict[action]): Dictionary containing action(s) to be executed
                (<span style="color:#C00000"><b>required</b></span>).

        Returns:
            (parallel,) dict[state], bool | 0 | 1 | 2, float: Dictionary containing next state(s)
            and auxiliary information, whether a terminal state is reached or 2 if the episode was
            aborted, observed reward, and parallel index vector in case of vectorized execution.
        """
        raise NotImplementedError

    def start_reset(self, num_parallel=None):
        if self._expect_receive is not None:
            raise TensorforceError.unexpected()
        self._expect_receive = 'reset'
        assert num_parallel is None or self.is_vectorizable()
        self._num_parallel = num_parallel

    def start_execute(self, actions):
        if self._expect_receive is not None:
            raise TensorforceError.unexpected()
        self._expect_receive = 'execute'
        assert self._actions is None
        self._actions = actions

    def receive_execute(self):
        if self._expect_receive == 'reset':
            self._expect_receive = None
            if self._num_parallel is None:
                states = self.reset()
            else:
                parallel, states = self.reset(num_parallel=num_parallel)
            if self._reset_output_check:
                self._check_states_output(states=states, function='reset')
                if self._num_parallel is not None:
                    TensorSpec(type='int', shape=(), num_values=self._num_parallel).np_assert(
                        x=parallel, batched=True,
                        message=(function + ': invalid {issue} for parallel.')
                    )
                self._reset_output_check = False
            if self._num_parallel is None:
                return states, -1, None
            else:
                return parallel, states, -1, None

        elif self._expect_receive == 'execute':
            self._expect_receive = None
            assert self._actions is not None
            if self._num_parallel is None:
                states, terminal, reward = self.execute(actions=self._actions)
            else:
                parallel, states, terminal, reward = self.execute(actions=self._actions)
            if self._execute_output_check:
                self._check_states_output(states=states, function='execute')
                if self._num_parallel is None:
                    if isinstance(reward, (np.generic, np.ndarray)):
                        reward = reward.item()
                    if isinstance(terminal, (np.generic, np.ndarray)):
                        terminal = terminal.item()
                    if not isinstance(terminal, bool) and \
                            (not isinstance(terminal, int) or terminal < 0 or terminal > 2):
                        raise TensorforceError(
                            'Environment.execute: invalid value {} for terminal.'.format(terminal)
                        )
                    if not isinstance(reward, (float, int)):
                        raise TensorforceError(
                            'Environment.execute: invalid type {} for reward.'.format(type(reward))
                        )
                else:
                    TensorSpec(type='int', shape=(), num_values=self._num_parallel).np_assert(
                        x=parallel, batched=True,
                        message=(function + ': invalid {issue} for parallel.')
                    )
                    TensorSpec(type='bool', shape=()).np_assert(
                        x=terminal, batched=True,
                        message=(function + ': invalid {issue} for terminal.')
                    )
                    TensorSpec(type='float', shape=()).np_assert(
                        x=reward, batched=True,
                        message=(function + ': invalid {issue} for reward.')
                    )
                self._execute_output_check = False
            self._actions = None
            if self._num_parallel is None:
                return states, int(terminal), reward
            else:
                return parallel, states, terminal, reward

        else:
            raise TensorforceError.unexpected()

    def _check_states_output(self, states, function):
        function = 'Environment.' + function
        states_spec = self.states()
        if 'type' in states_spec or 'shape' in states_spec:
            states_spec = TensorSpec(**states_spec)
            if isinstance(states, dict):
                for name in states:
                    if name != 'state' and not name.endswith('_mask'):
                        raise TensorforceError(function + ': invalid component {name} for state.')
                _states = states['state']
            else:
                _states = states
        else:
            states_spec = TensorsSpec(self.states())
            _states = dict()
            for name, state in states.items():
                if name in states_spec:
                    _states[name] = state
                elif not name.endswith('_mask'):
                    raise TensorforceError(function + ': invalid component {name} for state.')
        states_spec.np_assert(
            x=_states, batched=(self._num_parallel is not None),
            message=(function + ': invalid {issue} for {name} state.')
        )

class EnvironmentWrapper(Environment):

    def __init__(self, environment, max_episode_timesteps=None, reward_shaping=None):
        super().__init__()

        if isinstance(environment, EnvironmentWrapper):
            raise TensorforceError.unexpected()
        if environment.max_episode_timesteps() is not None and \
                max_episode_timesteps is not None and \
                environment.max_episode_timesteps() < max_episode_timesteps:
            raise TensorforceError.unexpected()

        self._environment = environment
        if max_episode_timesteps is None:
            self._max_episode_timesteps = self._environment.max_episode_timesteps()
        else:
            self._max_episode_timesteps = int(max_episode_timesteps)
            if self._environment.max_episode_timesteps() is None:
                self._environment.max_episode_timesteps = (lambda: int(max_episode_timesteps))
        self._timestep = None
        self._previous_states = None
        self._reward_shaping = reward_shaping

    def __str__(self):
        return str(self._environment)

    def states(self):
        return self._environment.states()

    def actions(self):
        return self._environment.actions()

    def max_episode_timesteps(self):
        return self._max_episode_timesteps

    def is_vectorizable(self):
        return self._environment.is_vectorizable()

    def close(self):
        return self._environment.close()

    def reset(self, num_parallel=None):
        self._timestep = 0
        assert num_parallel is None or self.is_vectorizable()
        self._num_parallel = num_parallel
        if self._num_parallel is None:
            states = self._environment.reset()
        else:
            parallel, states = self._environment.reset(num_parallel=self._num_parallel)
        if self._reset_output_check:
            self._check_states_output(states=states, function='reset')
            if self._num_parallel is not None:
                TensorSpec(type='int', shape=(), num_values=self._num_parallel).np_assert(
                    x=parallel, batched=True,
                    message=('Environment.reset: invalid {issue} for parallel.')
                )
            self._reset_output_check = False
        if self._reward_shaping is not None:
            self._previous_states = states
        if self._num_parallel is None:
            return states
        else:
            return parallel, states

    def execute(self, actions):
        if self._timestep is None:
            raise TensorforceError(
                message="An environment episode has to be initialized by calling reset() first."
            )
        assert self._max_episode_timesteps is None or self._timestep < self._max_episode_timesteps
        if self._num_parallel is None:
            states, terminal, reward = self._environment.execute(actions=actions)
        else:
            parallel, states, terminal, reward = self._environment.execute(actions=actions)
        if self._execute_output_check:
            self._check_states_output(states=states, function='execute')
            if self._num_parallel is None:
                if isinstance(reward, (np.generic, np.ndarray)):
                    reward = reward.item()
                if isinstance(terminal, (np.generic, np.ndarray)):
                    terminal = terminal.item()
                if not isinstance(terminal, bool) and \
                        (not isinstance(terminal, int) or terminal < 0 or terminal > 2):
                    raise TensorforceError(
                        'Environment.execute: invalid value {} for terminal.'.format(terminal)
                    )
                if not isinstance(reward, (float, int)):
                    raise TensorforceError(
                        'Environment.execute: invalid type {} for reward.'.format(type(reward))
                    )
            else:
                TensorSpec(type='int', shape=(), num_values=self._num_parallel).np_assert(
                    x=parallel, batched=True,
                    message='Environment.execute: invalid {issue} for parallel.'
                )
                TensorSpec(type='bool', shape=()).np_assert(
                    x=terminal, batched=True,
                    message='Environment.execute: invalid {issue} for terminal.'
                )
                TensorSpec(type='float', shape=()).np_assert(
                    x=reward, batched=True,
                    message='Environment.execute: invalid {issue} for reward.'
                )
            self._execute_output_check = False
        if self._reward_shaping is not None:
            if isinstance(self._reward_shaping, str):
                reward = eval(self._reward_shaping, dict(), dict(
                    states=self._previous_states, actions=actions, terminal=terminal, reward=reward,
                    next_states=states, math=math, np=np, random=random
                ))
            else:
                reward = self._reward_shaping(
                    self._previous_states, actions, terminal, reward, states
                )
            if isinstance(reward, tuple):
                reward, terminal = reward
            if isinstance(reward, (np.generic, np.ndarray)):
                reward = reward.item()
            if isinstance(terminal, (np.generic, np.ndarray)):
                terminal = terminal.item()
            self._previous_states = states
        self._timestep += 1
        if self._num_parallel is None:
            terminal = int(terminal)
            if terminal == 0 and self._max_episode_timesteps is not None and \
                    self._timestep >= self._max_episode_timesteps:
                terminal = 2
            if terminal > 0:
                self._timestep = None
            return states, terminal, reward
        else:
            terminal = terminal.astype(util.np_dtype('int'))
            if (terminal == 0).any() and self._max_episode_timesteps is not None and \
                    self._timestep >= self._max_episode_timesteps:
                terminal = np.where(terminal == 0, 2, terminal)
                parallel = parallel[:0]
                states = None
            if (terminal > 0).all():
                self._timestep = None
            return parallel, states, terminal, reward

    _ATTRIBUTES = frozenset([
        '_actions', 'create', '_environment', '_execute_output_check', '_expect_receive',
        '_previous_states', '_max_episode_timesteps', '_num_parallel', '_reset_output_check',
        '_reward_shaping', '_timestep'
    ])

    def __getattr__(self, name):
        if name in EnvironmentWrapper._ATTRIBUTES:
            return super().__getattr__(name)
        else:
            return getattr(self._environment, name)

    def __setattr__(self, name, value):
        if name in EnvironmentWrapper._ATTRIBUTES:
            super().__setattr__(name, value)
        else:
            return setattr(self._environment, name, value)


class RemoteEnvironment(Environment):

    @classmethod
    def proxy_send(cls, connection, function, **kwargs):
        raise NotImplementedError

    @classmethod
    def proxy_receive(cls, connection):
        raise NotImplementedError

    @classmethod
    def proxy_close(cls, connection):
        raise NotImplementedError

    @classmethod
    def remote_send(cls, connection, success, result):
        raise NotImplementedError

    @classmethod
    def remote_receive(cls, connection):
        raise NotImplementedError

    @classmethod
    def remote_close(cls, connection):
        raise NotImplementedError

    @classmethod
    def remote(
        cls, connection, environment, max_episode_timesteps=None, reward_shaping=None, **kwargs
    ):
        try:
            env = None
            env = Environment.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps,
                reward_shaping=reward_shaping, **kwargs
            )

            while True:
                attribute, kwargs = cls.remote_receive(connection=connection)

                if attribute in ('reset', 'execute'):
                    environment_start = time.time()

                try:
                    result = getattr(env, attribute)
                    if callable(result):
                        if kwargs is None:
                            result = None
                        else:
                            result = result(**kwargs)
                    elif kwargs is None:
                        pass
                    elif len(kwargs) == 1 and 'value' in kwargs:
                        setattr(env, attribute, kwargs['value'])
                        result = None
                    else:
                        raise TensorforceError(message="Invalid remote attribute/function access.")
                except AttributeError:
                    if kwargs is None or len(kwargs) != 1 or 'value' not in kwargs:
                        raise TensorforceError(message="Invalid remote attribute/function access.")
                    setattr(env, attribute, kwargs['value'])
                    result = None

                if attribute in ('reset', 'execute'):
                    seconds = time.time() - environment_start
                    if attribute == 'reset':
                        result = (result, seconds)
                    else:
                        result += (seconds,)

                cls.remote_send(connection=connection, success=True, result=result)

                if attribute == 'close':
                    break

        except BaseException:
            etype, value, traceback = sys.exc_info()
            cls.remote_send(
                connection=connection, success=False,
                result=(str(etype), str(value), format_tb(traceback))
            )

            try:
                if env is not None:
                    env.close()
            except BaseException:
                pass
            finally:
                etype, value, traceback = sys.exc_info()
                cls.remote_send(
                    connection=connection, success=False,
                    result=(str(etype), str(value), format_tb(traceback))
                )

        finally:
            cls.remote_close(connection=connection)

    def __init__(self, connection, blocking=False):
        super().__init__()
        self._connection = connection
        self._blocking = blocking
        self._observation = None
        self._thread = None
        self._episode_seconds = None

    def send(self, function, kwargs):
        if self._expect_receive is not None:
            assert function != 'close'
            self.close()
            raise TensorforceError.unexpected()
        self._expect_receive = function

        try:
            self.__class__.proxy_send(connection=self._connection, function=function, kwargs=kwargs)
        except BaseException:
            self.__class__.proxy_close(connection=self._connection)
            raise

    def receive(self, function):
        if self._expect_receive != function:
            assert function != 'close'
            self.close()
            raise TensorforceError.unexpected()
        self._expect_receive = None

        try:
            success, result = self.__class__.proxy_receive(connection=self._connection)
        except BaseException:
            self.__class__.proxy_close(connection=self._connection)
            raise

        if success:
            return result
        else:
            self.__class__.proxy_close(connection=self._connection)
            etype, value, traceback = result
            raise TensorforceError(message='\n{}\n{}: {}`'.format(''.join(traceback), etype, value))

    _ATTRIBUTES = frozenset([
        '_actions', '_blocking', '_connection', 'create', '_episode_seconds',
        '_execute_output_check', '_expect_receive', '_num_parallel', '_observation',
        '_reset_output_check', '_thread'
    ])

    def __getattr__(self, name):
        if name in RemoteEnvironment._ATTRIBUTES:
            return super().__getattr__(name)
        else:
            self.send(function=name, kwargs=None)
            result = self.receive(function=name)
            if result is None:
                def proxy_function(*args, **kwargs):
                    if len(args) > 0:
                        raise TensorforceError(
                            message="Remote environment function call requires keyword arguments."
                        )
                    self.send(function=name, kwargs=kwargs)
                    return self.receive(function=name)
                return proxy_function
            else:
                return result

    def __setattr__(self, name, value):
        if name in RemoteEnvironment._ATTRIBUTES:
            super().__setattr__(name, value)
        else:
            self.send(function=name, kwargs=dict(value=value))
            result = self.receive(function=name)
            assert result is None

    def __str__(self):
        self.send(function='__str__', kwargs=dict())
        return self.receive(function='__str__')

    def states(self):
        self.send(function='states', kwargs=dict())
        return self.receive(function='states')

    def actions(self):
        self.send(function='actions', kwargs=dict())
        return self.receive(function='actions')

    def max_episode_timesteps(self):
        self.send(function='max_episode_timesteps', kwargs=dict())
        return self.receive(function='max_episode_timesteps')

    def close(self):
        if self._thread is not None:
            self._thread.join()
        if self._expect_receive is not None:
            self.receive(function=self._expect_receive)
        self.send(function='close', kwargs=dict())
        self.receive(function='close')
        self.__class__.proxy_close(connection=self._connection)
        self._connection = None
        self._observation = None
        self._thread = None

    def reset(self):
        self._episode_seconds = 0.0
        self.send(function='reset', kwargs=dict())
        states, seconds = self.receive(function='reset')
        self._episode_seconds += seconds
        return states

    def execute(self, actions):
        self.send(function='execute', kwargs=dict(actions=actions))
        states, terminal, reward, seconds = self.receive(function='execute')
        self._episode_seconds += seconds
        return states, int(terminal), reward

    def start_reset(self):
        self._episode_seconds = 0.0
        if self._blocking:
            self.send(function='reset', kwargs=dict())
        else:
            if self._thread is not None:  # TODO: not expected
                self._thread.join()
            self._observation = None
            self._thread = Thread(target=self.finish_reset)
            self._thread.start()

    def finish_reset(self):
        assert self._thread is not None and self._observation is None
        self._observation = (self.reset(), -1, None)
        self._thread = None

    def start_execute(self, actions):
        if self._blocking:
            self.send(function='execute', kwargs=dict(actions=actions))
        else:
            assert self._thread is None and self._observation is None
            self._thread = Thread(target=self.finish_execute, kwargs=dict(actions=actions))
            self._thread.start()

    def finish_execute(self, actions):
        assert self._thread is not None and self._observation is None
        self._observation = self.execute(actions=actions)
        self._thread = None

    def receive_execute(self):
        if self._blocking:
            if self._expect_receive == 'reset':
                states, seconds = self.receive(function='reset')
                self._episode_seconds += seconds
                return states, -1, None
            else:
                states, terminal, reward, seconds = self.receive(function='execute')
                self._episode_seconds += seconds
                return states, int(terminal), reward
        else:
            if self._thread is not None:
                return None
            else:
                assert self._observation is not None
                observation = self._observation
                self._observation = None
                return observation
