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

import logging

from six.moves import xrange
import sys
import copy

from tensorforce.execution import SingleRunner, ThreadedRunner
from tensorforce.execution.threaded_runner import clone_worker_agent

logging.getLogger('tensorflow').disabled = True


class BaseTest(object):
    """
    Base class for tests of Agent functionality.
    """

    agent = None
    deterministic = None
    requires_network = True
    pass_threshold = 0.8

    def pre_run(self, agent, environment):
        """
        Called before `Runner.run`.
        """
        pass

    def base_test_pass(self, name, environment, network_spec, **kwargs):
        """
        Basic test loop, requires an Agent to achieve a certain performance on an environment.

        Args:
            name (str): The name of the test.
            environment (Environment): The Environment object to use for the test.
            network_spec (LayerBasedNetwork): The Network to use for the agent's model.
            kwargs (any):
                'multithreaded'=True if this run should be made using the ThreadedRunner.
        """
        sys.stdout.write('\n{} ({}):'.format(self.__class__.agent.__name__, name))
        sys.stdout.flush()

        threaded = kwargs.pop("multithreaded", False)

        passed = 0
        for _ in xrange(3):
            if self.__class__.requires_network:
                agent = self.__class__.agent(
                    states_spec=environment.states,
                    actions_spec=environment.actions,
                    network_spec=network_spec,
                    **kwargs
                )
            else:
                agent = self.__class__.agent(
                    states_spec=environment.states,
                    actions_spec=environment.actions,
                    **kwargs
                )

            if threaded:
                num_workers = 5
                agents = clone_worker_agent(agent, num_workers, environment,
                                                network_spec if self.__class__.requires_network else None, kwargs)
                environments = [environment]
                for _ in range(num_workers - 1):
                    environments.append(copy.deepcopy(environment))
                runner = ThreadedRunner(agent=agents, environment=environments)
            else:
                runner = SingleRunner(agent=agent, environment=environment)
                self.pre_run(agent=agent, environment=environment)

            def episode_finished(r, worker_id=0):
                episodes_passed = [
                    rw / ln >= self.__class__.pass_threshold
                    for rw, ln in zip(r.episode_rewards[-100:], r.episode_timesteps[-100:])
                ]
                return r.episode < 100 or not all(episodes_passed)

            runner.run(episodes=3000, deterministic=self.__class__.deterministic, episode_finished=episode_finished)
            runner.close()

            sys.stdout.write(' ' + str(runner.episode))
            sys.stdout.flush()
            if all(rw / ln >= self.__class__.pass_threshold
                    for rw, ln in zip(runner.episode_rewards[-100:], runner.episode_timesteps[-100:])):
                passed += 1
            if passed == 2:
                break

        sys.stdout.write(' ==> {} passed\n'.format(passed))
        sys.stdout.flush()
        self.assertTrue(passed >= 2)

    def base_test_run(self, name, environment, network_spec, **kwargs):
        """
        Run test, tests whether algorithm can run and update without compilation errors,
        not whether it passes.

        Args:
            name (str): The name of the test.
            environment (Environment): The Environment object to use for the test.
            network_spec (LayerBasedNetwork): The Network to use for the agent's model.
            kwargs (any):
                'multithreaded'=True if this run should be made using the ThreadedRunner.
        """

        sys.stdout.write('\n{} ({}):'.format(self.__class__.agent.__name__, name))
        sys.stdout.flush()

        threaded = kwargs.pop("multithreaded", False)

        if self.__class__.requires_network:
            agent = self.__class__.agent(
                states_spec=environment.states,
                actions_spec=environment.actions,
                network_spec=network_spec,
                **kwargs
            )
        else:
            agent = self.__class__.agent(
                states_spec=environment.states,
                actions_spec=environment.actions,
                **kwargs
            )

        if threaded:
            num_workers = 5
            agents = duplicate_worker_agent(agent, num_workers, environment,
                                            network_spec if self.__class__.requires_network else None, kwargs)
            environments = [environment]
            for _ in range(num_workers - 1):
                environments.append(copy.deepcopy(environment))
            runner = ThreadedRunner(agent=agents, environment=environments)
        else:
            runner = SingleRunner(agent=agent, environment=environment)
            self.pre_run(agent=agent, environment=environment)

        def episode_finished(r, worker_id=0):
            episodes_passed = [
                rw / ln >= self.__class__.pass_threshold
                for rw, ln in zip(r.episode_rewards[-100:], r.episode_timesteps[-100:])
            ]
            return r.episode < 100 or not all(episodes_passed)

        runner.run(num_episodes=100, deterministic=self.__class__.deterministic, episode_finished=episode_finished)
        runner.close()

        sys.stdout.write('==> {} ran\n'.format(1))
        sys.stdout.flush()

