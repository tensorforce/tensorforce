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

from datetime import datetime
import logging
from random import randint
import sys

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from test.unittest_environment import UnittestEnvironment


logging.getLogger('tensorflow').disabled = True


class UnittestBase(object):
    """
    Unit-test base class.
    """

    # Agent config.
    config = dict(
        update=4, network=dict(type='auto', size=8, internal_rnn=2), objective='policy_gradient',
        reward_estimation=dict(horizon=2)
    )

    # Flags for exclusion of action types.
    exclude_bool_action = False
    exclude_int_action = False
    exclude_float_action = False
    exclude_bounded_action = False

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
        self, states=None, actions=None, environment=None, timestep_range=None, action_masks=True,
        exclude_bool_action=False, exclude_int_action=False, exclude_float_action=False,
        exclude_bounded_action=False, **config
    ):
        """
        Generic unit-test preparation.
        """
        if environment is None:
            if states is None:
                states = dict(
                    bool_state=dict(type='bool', shape=(1,)),
                    int_state=dict(type='int', shape=(2,), num_values=4),
                    float_state=dict(type='float', shape=(1, 1, 2)),
                    bounded_state=dict(type='float', shape=(), min_value=-0.5, max_value=0.5)
                )

            if actions is None:
                actions = dict()
                if not exclude_bool_action and not self.__class__.exclude_bool_action:
                    actions['bool_action'] = dict(type='bool', shape=(1,))
                if not exclude_int_action and not self.__class__.exclude_int_action:
                    actions['int_action'] = dict(type='int', shape=(2,), num_values=4)
                if not exclude_float_action and not self.__class__.exclude_float_action:
                    actions['float_action'] = dict(type='float', shape=(1, 1))
                if not exclude_bounded_action and not self.__class__.exclude_bounded_action:
                    actions['bounded_action'] = dict(
                        type='float', shape=(2,), min_value=-0.5, max_value=0.5
                    )

            if timestep_range is None:
                environment = UnittestEnvironment(
                    states=states, actions=actions, timestep_range=(1, 5),
                    action_masks=action_masks
                )
            else:
                environment = UnittestEnvironment(
                    states=states, actions=actions, timestep_range=timestep_range,
                    action_masks=action_masks
                )

        else:
            self.assertTrue(expr=(timestep_range is None))

        environment = Environment.create(environment=environment)

        agent = Agent.create(agent=config, environment=environment, **self.__class__.config)

        return agent, environment

    def unittest(
        self, states=None, actions=None, environment=None, timestep_range=None, action_masks=True,
        num_episodes=None, **config
    ):
        """
        Generic unit-test.
        """
        agent, environment = self.prepare(
            states=states, actions=actions, environment=environment, timestep_range=timestep_range,
            action_masks=action_masks, **config
        )

        self.runner = Runner(agent=agent, environment=environment)
        if num_episodes is None:
            self.runner.run(num_episodes=randint(5, 10), use_tqdm=False)
        else:
            self.runner.run(num_episodes=num_episodes, use_tqdm=False)
        self.runner.close()

        self.finished_test()
