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

import os
from collections import OrderedDict

import numpy as np

from tensorforce import TensorforceError, util
from tensorforce.core import ArrayDict, ListDict, TensorSpec, TensorsSpec


class Recorder(object):
    """
    Recorder wrapper (specification key: `recorder`).

    Args:
        fn_act (lambda[states -> actions]): Act-function mapping states to actions which is supposed
            to be recorded.
    """

    def __init__(
        self, fn_act, states, actions, max_episode_timesteps=None, parallel_interactions=1,
        recorder=None
    ):
        self.is_initialized = False

        # fn_act=None means Agent
        if fn_act is None:
            from tensorforce import Agent
            assert isinstance(self, Agent)
            self._is_agent = True
        else:
            self._is_agent = False
            self.fn_act = fn_act

        # States/actions, plus single state/action flag
        if 'type' in states or 'shape' in states:
            self.states_spec = TensorsSpec(singleton=states)
        else:
            self.states_spec = TensorsSpec(states)
        if 'type' in actions or 'shape' in actions:
            self.actions_spec = TensorsSpec(singleton=actions)
        else:
            self.actions_spec = TensorsSpec(actions)

        # Max episode timesteps
        self.max_episode_timesteps = max_episode_timesteps

        # Parallel interactions
        if isinstance(parallel_interactions, int):
            if parallel_interactions <= 0:
                raise TensorforceError.value(
                    name='Agent', argument='parallel_interactions', value=parallel_interactions,
                    hint='<= 0'
                )
            self.parallel_interactions = parallel_interactions
        else:
            raise TensorforceError.type(
                name='Agent', argument='parallel_interactions', dtype=type(parallel_interactions)
            )

        # Other specifications
        self.internals_spec = TensorsSpec()
        self.terminal_spec = TensorSpec(type=int, shape=(), num_values=3)
        self.reward_spec = TensorSpec(type=float, shape=())
        self.parallel_spec = TensorSpec(type=int, shape=(), num_values=self.parallel_interactions)

        # Recorder
        if isinstance(recorder, str):
            recorder = dict(directory=recorder)
        if recorder is None:
            pass
        elif not all(key in ('directory', 'frequency', 'max-traces', 'start') for key in recorder):
            raise TensorforceError.value(
                name='Agent', argument='recorder values', value=list(recorder),
                hint='not from {directory,frequency,max-traces,start}'
            )
        self.recorder = recorder if recorder is None else dict(recorder)

    def initialize(self):
        # Check whether already initialized
        if self.is_initialized:
            raise TensorforceError(
                message="Agent is already initialized, possibly as part of Agent.create()."
            )
        self.is_initialized = True

        # Act-observe timestep check
        self.timestep_counter = np.zeros(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='int')
        )
        self.timestep_completed = np.ones(
            shape=(self.parallel_interactions,), dtype=util.np_dtype(dtype='bool')
        )

        # Recorder buffers if required
        if self.recorder is not None:
            self.num_episodes = 0

            self.buffers = ListDict()
            self.buffers['terminal'] = [list() for _ in range(self.parallel_interactions)]
            self.buffers['reward'] = [list() for _ in range(self.parallel_interactions)]

            def function(spec):
                return [list() for _ in range(self.parallel_interactions)]

            self.buffers['states'] = self.states_spec.fmap(function=function, cls=ListDict)
            self.buffers['actions'] = self.actions_spec.fmap(function=function, cls=ListDict)

            function = (lambda x: list())

            self.recorded = ListDict()
            self.recorded['states'] = self.states_spec.fmap(function=function, cls=ListDict)
            self.recorded['actions'] = self.actions_spec.fmap(function=function, cls=ListDict)
            self.recorded['terminal'] = list()
            self.recorded['reward'] = list()

    def close(self):
        pass

    def reset(self):
        # Reset timestep check
        self.timestep_counter[:] = 0
        self.timestep_completed[:] = True

        # Reset buffers
        if self.recorder is not None:
            for buffer in self.buffers.values():
                for x in buffer:
                    x.clear()
            if self.recorder is not None:
                for x in self.recorded.values():
                    x.clear()

    def initial_internals(self):
        return OrderedDict()

    def act(
        self, states, internals=None, parallel=0, independent=False, deterministic=False, **kwargs
    ):
        # Independent and internals
        is_internals_none = (internals is None)
        if independent:
            if parallel != 0:
                raise TensorforceError.invalid(
                    name='Agent.act', argument='parallel', condition='independent is true'
                )
            if is_internals_none and len(self.internals_spec) > 0:
                raise TensorforceError.required(
                    name='Agent.act', argument='internals', condition='independent is true'
                )
        else:
            if not is_internals_none:
                raise TensorforceError.invalid(
                    name='Agent.act', argument='internals', condition='independent is false'
                )

        # Independent and deterministic
        if deterministic and not independent:
            raise TensorforceError.invalid(
                name='Agent.act', argument='deterministic', condition='independent is false'
            )

        # Process states input and infer batching structure
        states, batched, num_parallel, is_iter_of_dicts, input_type = self._process_states_input(
            states=states, function_name='Agent.act'
        )

        if independent:
            # Independent mode: handle internals argument
            if is_internals_none:
                # Default input internals=None
                pass

            elif is_iter_of_dicts:
                # Input structure iter[dict[internal]]
                if not isinstance(internals, (tuple, list)):
                    raise TensorforceError.type(
                        name='Agent.act', argument='internals', dtype=type(internals),
                        hint='is not tuple/list'
                    )
                internals = [ArrayDict(internal) for internal in internals]
                internals = internals[0].fmap(
                    function=(lambda *xs: np.stack(xs, axis=0)), zip_values=internals[1:]
                )

            else:
                # Input structure dict[iter[internal]]
                if not isinstance(internals, dict):
                    raise TensorforceError.type(
                        name='Agent.act', argument='internals', dtype=type(internals),
                        hint='is not dict'
                    )
                internals = ArrayDict(internals)

            if not independent or not is_internals_none:
                # Expand inputs if not batched
                if not batched:
                    internals = internals.fmap(function=(lambda x: np.expand_dims(x, axis=0)))

                # Check number of inputs
                for name, internal in internals.items():
                    if internal.shape[0] != num_parallel:
                        raise TensorforceError.value(
                            name='Agent.act', argument='len(internals[{}])'.format(name),
                            value=internal.shape[0], hint='!= len(states)'
                        )

        else:
            # Non-independent mode: handle parallel input
            if parallel == 0:
                # Default input parallel=0
                if batched:
                    assert num_parallel == self.parallel_interactions
                    parallel = np.asarray(list(range(num_parallel)))
                else:
                    parallel = np.asarray([parallel])

            elif batched:
                # Batched input
                parallel = np.asarray(parallel)

            else:
                # Expand input if not batched
                parallel = np.asarray([parallel])

            # Check number of inputs
            if parallel.shape[0] != num_parallel:
                raise TensorforceError.value(
                    name='Agent.act', argument='len(parallel)', value=len(parallel),
                    hint='!= len(states)'
                )

        # If not independent, check whether previous timesteps were completed
        if not independent:
            if not self.timestep_completed[parallel].all():
                raise TensorforceError(
                    message="Calling agent.act must be preceded by agent.observe."
                )
            self.timestep_completed[parallel] = False

        # Buffer inputs for recording
        if self.recorder is not None and not independent and \
                self.num_episodes >= self.recorder.get('start', 0):
            for n in range(num_parallel):
                for name in self.states_spec:
                    self.buffers['states'][name][parallel[n]].append(states[name][n])

        # fn_act()
        if self._is_agent:
            actions, internals = self.fn_act(
                states=states, internals=internals, parallel=parallel, independent=independent,
                deterministic=deterministic, is_internals_none=is_internals_none,
                num_parallel=num_parallel
            )
        else:
            if batched:
                assert False
            else:
                states = states.fmap(function=(lambda x: x[0].item() if x.shape == (1,) else x[0]))
                actions = self.fn_act(states.to_kwargs())
                if self.actions_spec.is_singleton():
                    actions = ArrayDict(singleton=np.asarray([actions]))
                else:
                    actions = ArrayDict(actions)
                    actions = actions.fmap(function=(lambda x: np.asarray([x])))

        # Buffer outputs for recording
        if self.recorder is not None and not independent and \
                self.num_episodes >= self.recorder.get('start', 0):
            for n in range(num_parallel):
                for name in self.actions_spec:
                    self.buffers['actions'][name][parallel[n]].append(actions[name][n])

        # Unbatch actions
        if batched:
            # If inputs were batched, turn list of dicts into dict of lists
            function = (lambda x: x.item() if x.shape == () else x)
            # TODO: recursive
            if self.actions_spec.is_singleton():
                actions = actions.singleton()
                actions = input_type(function(actions[n]) for n in range(num_parallel))
            else:
                actions = input_type(
                    OrderedDict(((name, function(x[n])) for name, x in actions.items()))
                    for n in range(num_parallel)
                )

            if independent and not is_internals_none and is_iter_of_dicts:
                # TODO: recursive
                internals = input_type(
                    OrderedDict(((name, function(x[n])) for name, x in internals.items()))
                    for n in range(num_parallel)
                )

        else:
            # If inputs were not batched, unbatch outputs
            function = (lambda x: x.item() if x.shape == (1,) else x[0])
            if self.actions_spec.is_singleton():
                actions = function(actions.singleton())
            else:
                actions = actions.fmap(function=function, cls=OrderedDict)
            if independent and not is_internals_none:
                internals = internals.fmap(function=function, cls=OrderedDict)

        if independent and not is_internals_none:
            return actions, internals
        else:
            return actions

    def observe(self, reward=0.0, terminal=False, parallel=0):
        # Check whether inputs are batched
        if util.is_iterable(x=reward):
            reward = np.asarray(reward)
            num_parallel = reward.shape[0]
            if terminal is False:
                terminal = np.asarray([0 for _ in range(num_parallel)])
            else:
                terminal = np.asarray(terminal)
            if parallel == 0:
                assert num_parallel == self.parallel_interactions
                parallel = np.asarray(list(range(num_parallel)))
            else:
                parallel = np.asarray(parallel)

        elif util.is_iterable(x=terminal):
            terminal = np.asarray([int(t) for t in terminal])
            num_parallel = terminal.shape[0]
            if reward == 0.0:
                reward = np.asarray([0.0 for _ in range(num_parallel)])
            else:
                reward = np.asarray(reward)
            if parallel == 0:
                assert num_parallel == self.parallel_interactions
                parallel = np.asarray(list(range(num_parallel)))
            else:
                parallel = np.asarray(parallel)

        elif util.is_iterable(x=parallel):
            parallel = np.asarray(parallel)
            num_parallel = parallel.shape[0]
            if reward == 0.0:
                reward = np.asarray([0.0 for _ in range(num_parallel)])
            else:
                reward = np.asarray(reward)
            if terminal is False:
                terminal = np.asarray([0 for _ in range(num_parallel)])
            else:
                terminal = np.asarray(terminal)

        else:
            reward = np.asarray([float(reward)])
            terminal = np.asarray([int(terminal)])
            parallel = np.asarray([int(parallel)])
            num_parallel = 1

        # Check whether shapes/lengths are consistent
        if parallel.shape[0] == 0:
            raise TensorforceError.value(
                name='Agent.observe', argument='len(parallel)', value=parallel.shape[0], hint='= 0'
            )
        if reward.shape != parallel.shape:
            raise TensorforceError.value(
                name='Agent.observe', argument='len(reward)', value=reward.shape,
                hint='!= parallel length'
            )
        if terminal.shape != parallel.shape:
            raise TensorforceError.value(
                name='Agent.observe', argument='len(terminal)', value=terminal.shape,
                hint='!= parallel length'
            )

        # Convert terminal to int if necessary
        if terminal.dtype is util.np_dtype(dtype='bool'):
            zeros = np.zeros_like(terminal, dtype=util.np_dtype(dtype='int'))
            ones = np.ones_like(terminal, dtype=util.np_dtype(dtype='int'))
            terminal = np.where(terminal, ones, zeros)

        # Check whether current timesteps are not completed
        if self.timestep_completed[parallel].any():
            raise TensorforceError(message="Calling agent.observe must be preceded by agent.act.")
        self.timestep_completed[parallel] = True

        # Check whether episode is too long
        self.timestep_counter[parallel] += 1
        if self.max_episode_timesteps is not None and np.logical_and(
            terminal == 0, self.timestep_counter[parallel] > self.max_episode_timesteps
        ).any():
            raise TensorforceError(message="Episode longer than max_episode_timesteps.")
        self.timestep_counter[parallel] = np.where(terminal > 0, 0, self.timestep_counter[parallel])

        if self.recorder is None:
            pass

        elif self.num_episodes < self.recorder.get('start', 0):
            # Increment num_episodes
            for t in terminal.tolist():
                if t > 0:
                    self.num_episodes += 1

        else:
            # Store values per parallel interaction
            for p, t, r in zip(parallel.tolist(), terminal.tolist(), reward.tolist()):

                # Buffer inputs
                self.buffers['terminal'][p].append(t)
                self.buffers['reward'][p].append(r)

                # Continue if not terminal
                if t == 0:
                    continue
                self.num_episodes += 1

                # Buffered terminal/reward inputs
                for name in self.states_spec:
                    self.recorded['states'][name].append(
                        np.stack(self.buffers['states'][name][p], axis=0)
                    )
                    self.buffers['states'][name][p].clear()
                for name, spec in self.actions_spec.items():
                    self.recorded['actions'][name].append(
                        np.stack(self.buffers['actions'][name][p], axis=0)
                    )
                    self.buffers['actions'][name][p].clear()
                self.recorded['terminal'].append(
                    np.array(self.buffers['terminal'][p], dtype=self.terminal_spec.np_type())
                )
                self.buffers['terminal'][p].clear()
                self.recorded['reward'].append(
                    np.array(self.buffers['reward'][p], dtype=self.reward_spec.np_type())
                )
                self.buffers['reward'][p].clear()

                # Check whether recording step
                if (self.num_episodes - self.recorder.get('start', 0)) \
                        % self.recorder.get('frequency', 1) != 0:
                    continue

                # Manage recorder directory
                directory = self.recorder['directory']
                if os.path.isdir(directory):
                    files = sorted(
                        f for f in os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, f))
                        and os.path.splitext(f)[1] == '.npz'
                    )
                else:
                    os.makedirs(directory)
                    files = list()
                max_traces = self.recorder.get('max-traces')
                if max_traces is not None and len(files) > max_traces - 1:
                    for filename in files[:-max_traces + 1]:
                        filename = os.path.join(directory, filename)
                        os.remove(filename)

                # Write recording file
                filename = os.path.join(directory, 'trace-{:09d}.npz'.format(self.num_episodes - 1))
                # time.strftime('%Y%m%d-%H%M%S')
                kwargs = self.recorded.fmap(function=np.concatenate, cls=ArrayDict).items()
                np.savez_compressed(file=filename, **dict(kwargs))

                # Clear recorded values
                for recorded in self.recorded.values():
                    recorded.clear()

        if self._is_agent:
            return reward, terminal, parallel
        else:
            return 0

    def _process_states_input(self, states, function_name):
        if self.states_spec.is_singleton() and not isinstance(states, dict) and not (
            util.is_iterable(x=states) and isinstance(states[0], dict)
        ):
            # Single state
            input_type = type(states)
            states = np.asarray(states)

            if states.shape == self.states_spec.value().shape:
                # Single state is not batched
                states = ArrayDict(singleton=np.expand_dims(states, axis=0))
                batched = False
                num_instances = 1
                is_iter_of_dicts = None
                input_type = None

            else:
                # Single state is batched, iter[state]
                assert states.shape[1:] == self.states_spec.value().shape
                assert input_type in (tuple, list, np.ndarray)
                num_instances = states.shape[0]
                states = ArrayDict(singleton=states)
                batched = True
                is_iter_of_dicts = True  # Default

        elif util.is_iterable(x=states):
            # States is batched, iter[dict[state]]
            batched = True
            num_instances = len(states)
            is_iter_of_dicts = True
            input_type = type(states)
            assert input_type in (tuple, list)
            if num_instances == 0:
                raise TensorforceError.value(
                    name=function_name, argument='len(states)', value=num_instances, hint='= 0'
                )
            for n, state in enumerate(states):
                if not isinstance(state, dict):
                    raise TensorforceError.type(
                        name=function_name, argument='states[{}]'.format(n), dtype=type(state),
                        hint='is not dict'
                    )
            # Turn iter of dicts into dict of arrays
            # (Doesn't use self.states_spec since states also contains auxiliaries)
            states = [ArrayDict(state) for state in states]
            states = states[0].fmap(
                function=(lambda *xs: np.stack(xs, axis=0)), zip_values=states[1:]
            )

        elif isinstance(states, dict):
            # States is dict, turn into arrays
            some_state = next(iter(states.values()))
            input_type = type(some_state)

            states = ArrayDict(states)

            name, spec = self.states_spec.item()
            if name is None:
                name = 'state'

            if states[name].shape == spec.shape:
                # States is not batched, dict[state]
                states = states.fmap(function=(lambda state: np.expand_dims(state, axis=0)))
                batched = False
                num_instances = 1
                is_iter_of_dicts = None
                input_type = None

            else:
                # States is batched, dict[iter[state]]
                assert states[name].shape[1:] == spec.shape
                assert input_type in (tuple, list, np.ndarray)
                batched = True
                num_instances = states[name].shape[0]
                is_iter_of_dicts = False
                if num_instances == 0:
                    raise TensorforceError.value(
                        name=function_name, argument='len(states)', value=num_instances, hint='= 0'
                    )

        else:
            raise TensorforceError.type(
                name=function_name, argument='states', dtype=type(states),
                hint='is not array/tuple/list/dict'
            )

        # Check number of inputs
        if any(state.shape[0] != num_instances for state in states.values()):
            raise TensorforceError.value(
                name=function_name, argument='len(states)',
                value=[state.shape[0] for state in states.values()], hint='inconsistent'
            )

        return states, batched, num_instances, is_iter_of_dicts, input_type
