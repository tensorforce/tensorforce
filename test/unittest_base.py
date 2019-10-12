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

from copy import deepcopy
from datetime import datetime
import logging
import sys

from tensorforce.agents import Agent
from tensorforce.core.layers import Layer
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from test.unittest_environment import UnittestEnvironment


logging.getLogger('tensorflow').disabled = True


class UnittestBase(object):
    """
    Unit-test base class.
    """

    # Unittest
    num_updates = None
    num_episodes = None
    num_timesteps = None

    # Environment
    timestep_range = (1, 5)
    states = dict(
        bool_state=dict(type='bool', shape=(1,)),
        int_state=dict(type='int', shape=(2,), num_values=4),
        float_state=dict(type='float', shape=(1, 1, 2)),
        bounded_state=dict(type='float', shape=(), min_value=-0.5, max_value=0.5)
    )
    actions = dict(
        bool_action=dict(type='bool', shape=(1,)),
        int_action=dict(type='int', shape=(2,), num_values=4),
        float_action=dict(type='float', shape=(1, 1)),
        bounded_action=dict(type='float', shape=(2,), min_value=-0.5, max_value=0.5)
    )

    # Exclude action types
    exclude_bool_action = False
    exclude_int_action = False
    exclude_float_action = False
    exclude_bounded_action = False

    # Agent
    agent = dict(
        update=4, policy=dict(network=dict(type='auto', size=8, internal_rnn=2)),
        objective='policy_gradient', reward_estimation=dict(horizon=2)
    )

    # Tensorforce config
    require_observe = False
    require_all = False

    def start_tests(self, name=None):
        """
        Start unit-test method.
        """
        if name is None:
            sys.stdout.write('\n{} {}: '.format(
                datetime.now().strftime('%H:%M:%S'), self.__class__.__name__[4:]
            ))
        else:
            sys.stdout.write('\n{} {} ({}): '.format(
                datetime.now().strftime('%H:%M:%S'), self.__class__.__name__[4:], name
            ))
        sys.stdout.flush()

    def finished_test(self, assertion=None):
        """
        Finished unit-test.
        """
        if assertion is None:
            assertion = True
        else:
            self.assertTrue(expr=assertion)
        if assertion:
            sys.stdout.write('.')
            sys.stdout.flush()

    def prepare(
        self, environment=None, timestep_range=None, states=None, actions=None,
        exclude_bool_action=False, exclude_int_action=False, exclude_float_action=False,
        exclude_bounded_action=False, require_observe=False, require_all=False, **agent
    ):
        """
        Generic unit-test preparation.
        """
        Layer.layers = None

        if environment is None:
            if states is None:
                states = deepcopy(self.__class__.states)

            if actions is None:
                actions = deepcopy(self.__class__.actions)
                if exclude_bool_action or self.__class__.exclude_bool_action:
                    actions.pop('bool_action')
                if exclude_int_action or self.__class__.exclude_int_action:
                    actions.pop('int_action')
                if exclude_float_action or self.__class__.exclude_float_action:
                    actions.pop('float_action')
                if exclude_bounded_action or self.__class__.exclude_bounded_action:
                    actions.pop('bounded_action')

            if timestep_range is None:
                timestep_range = self.__class__.timestep_range

            environment = UnittestEnvironment(
                states=states, actions=actions, timestep_range=timestep_range,
            )

        elif timestep_range is not None:
            raise TensorforceError.unexpected()

        environment = Environment.create(environment=environment)

        for key, value in self.__class__.agent.items():
            if key not in agent:
                agent[key] = value

        if self.__class__.require_all or require_all:
            config = None
        elif self.__class__.require_observe or require_observe:
            config = dict(api_functions=['reset', 'act', 'observe'])
        else:
            config = dict(api_functions=['reset', 'act'])

        agent = Agent.create(agent=agent, environment=environment, config=config)

        return agent, environment

    def unittest(
        self, num_updates=None, num_episodes=None, num_timesteps=None, environment=None,
        timestep_range=None, states=None, actions=None, exclude_bool_action=False,
        exclude_int_action=False, exclude_float_action=False, exclude_bounded_action=False,
        require_observe=False, require_all=False, **agent
    ):
        """
        Generic unit-test.
        """
        agent, environment = self.prepare(
            environment=environment, timestep_range=timestep_range, states=states, actions=actions,
            exclude_bool_action=exclude_bool_action, exclude_int_action=exclude_int_action,
            exclude_float_action=exclude_float_action,
            exclude_bounded_action=exclude_bounded_action, require_observe=require_observe,
            require_all=require_all, **agent
        )

        self.runner = Runner(agent=agent, environment=environment)

        assert (num_updates is not None) + (num_episodes is not None) + \
            (num_timesteps is not None) <= 1
        if num_updates is None and num_episodes is None and num_timesteps is None:
            num_updates = self.__class__.num_updates
            num_episodes = self.__class__.num_episodes
            num_timesteps = self.__class__.num_timesteps
        if num_updates is None and num_episodes is None and num_timesteps is None:
            num_updates = 2
        assert (num_updates is not None) + (num_episodes is not None) + \
            (num_timesteps is not None) == 1

        evaluation = not any([
            require_all, require_observe, self.__class__.require_all,
            self.__class__.require_observe
        ])
        self.runner.run(
            num_episodes=num_episodes, num_timesteps=num_timesteps, num_updates=num_updates,
            max_episode_timesteps=agent.max_episode_timesteps, use_tqdm=False,
            evaluation=evaluation
        )
        self.runner.close()

        self.finished_test()
