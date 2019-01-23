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
from random import randint
import sys

from tensorforce.execution import Runner
from tensorforce.tests.unittest_environment import UnittestEnvironment


logging.getLogger('tensorflow').disabled = True


class UnittestBase(object):
    """
    Unit-test base class.
    """

    agent = None
    config = None
    ignore_network = False

    def prepare(self, name, states, actions, agent=None, network=None, **kwargs):
        """
        Generic unit-test preparation.
        """
        if agent is None:
            agent = self.__class__.agent

        sys.stdout.write('\n{} ({}):\n'.format(agent.__name__, name))
        sys.stdout.flush()

        environment = UnittestEnvironment(states=states, actions=actions)

        if self.__class__.config is not None:
            for key, arg in self.__class__.config.items():
                if key not in kwargs:
                    kwargs[key] = arg
        kwargs['states'] = environment.states()
        kwargs['actions'] = environment.actions()
        if not self.__class__.ignore_network and network is not None:
            kwargs['network'] = network

        agent = agent(**kwargs)

        return agent, environment

    def unittest(self, name, states, actions, agent=None, network=None, **kwargs):
        """
        Generic unit-test.
        """
        agent, environment = self.prepare(
            name=name, states=states, actions=actions, agent=agent, network=network, **kwargs
        )

        runner = Runner(agent=agent, environment=environment)
        runner.run(num_episodes=randint(5, 10))
        runner.close()

        sys.stdout.flush()
        self.assertTrue(expr=True)
