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

    def prepare(self, name, states, actions, network=None):
        """
        Generic unit-test preparation.
        """
        sys.stdout.write('\n{} ({}):\n'.format(self.__class__.agent.__name__, name))
        sys.stdout.flush()

        environment = UnittestEnvironment(states=states, actions=actions)

        if self.__class__.config is None:
            config = dict()
        config['states'] = environment.states
        config['actions'] = environment.actions
        if not self.__class__.ignore_network and network is not None:
            config['network'] = network

        agent = self.__class__.agent(**config)

        return agent, environment

    def unittest(self, name, states, actions, network=None):
        """
        Generic unit-test.
        """
        try:
            agent, environment = self.prepare(
                name=name, states=states, actions=actions, network=network
            )

            runner = Runner(agent=agent, environment=environment)
            runner.run(num_episodes=randint(2, 5))
            runner.close()

            sys.stdout.flush()
            self.assertTrue(expr=True)

        except Exception as exc:
            sys.stdout.write(str(exc))
            sys.stdout.flush()
            raise exc
            self.assertTrue(expr=False)
