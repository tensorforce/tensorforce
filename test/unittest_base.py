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

from copy import deepcopy
from datetime import datetime
import sys

from tensorforce import Agent, Environment, Runner
from test.unittest_environment import UnittestEnvironment


class UnittestBase(object):
    """
    Unit-test base class.
    """

    # Environment
    states = dict(
        bool_state=dict(type='bool', shape=(1,)),
        int_state=dict(type='int', shape=(1, 2), num_values=4),
        float_state=dict(type='float', shape=(), min_value=1.0, max_value=2.0)
    )
    actions = dict(
        bool_action=dict(type='bool', shape=(1,)),
        int_action=dict(type='int', shape=(2,), num_values=4),
        gaussian_action1=dict(type='float', shape=(1, 2), min_value=1.0, max_value=2.0),
        gaussian_action2=dict(type='float', shape=(), min_value=-2.0, max_value=1.0),
        beta_action=dict(type='float', shape=(), min_value=1.0, max_value=2.0)
    )
    min_timesteps = 5
    max_episode_timesteps = 10
    experience_update = True

    # Agent
    agent = dict(
        # Also used in: text_reward_estimation
        policy=dict(network=dict(type='auto', size=8, depth=1, rnn=2), distributions=dict(
            gaussian_action2=dict(type='gaussian', global_stddev=True), beta_action='beta'
        )), update=4, optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(
            horizon=3, estimate_advantage=True, predict_horizon_values='late',
            return_processing=dict(type='clipping', lower=-1.0, upper=1.0),
            advantage_processing='batch_normalization'
        ), baseline=dict(network=dict(type='auto', size=7, depth=1, rnn=1)),
        baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3),
        baseline_objective='state_value', l2_regularization=0.01, entropy_regularization=0.01,
        state_preprocessing='linear_normalization',
        reward_preprocessing=dict(type='clipping', lower=-1.0, upper=1.0),
        exploration=0.01, variable_noise=0.01,
        # Config default changes need to be adapted everywhere (search "config=dict"):
        #   test_agents, test_environments, test_examples, test_layers, test_precision,
        #   test_reward_estimation, test_saving, test_seed, test_summaries
        config=dict(eager_mode=True, create_debug_assertions=True, tf_log_level=20)
    )

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

    def environment_spec(self, states=None, actions=None):
        if states is None:
            states = deepcopy(self.__class__.states)

        if actions is None:
            actions = deepcopy(self.__class__.actions)

        return dict(
            environment=UnittestEnvironment,
            max_episode_timesteps=self.__class__.max_episode_timesteps,
            states=states, actions=actions, min_timesteps=self.__class__.min_timesteps
        )

    def agent_spec(self, **agent):
        for key, value in self.__class__.agent.items():
            if key not in agent:
                agent[key] = value

        return dict(agent=agent)

    def prepare(self, environment=None, states=None, actions=None, **agent):
        """
        Generic unit-test preparation.
        """
        if environment is None:
            environment = self.environment_spec(states=states, actions=actions)
            environment = Environment.create(environment=environment)

        else:
            environment = Environment.create(
                environment=environment, max_episode_timesteps=self.__class__.max_episode_timesteps
            )

        agent = self.agent_spec(**agent)

        agent = Agent.create(agent=agent, environment=environment)

        return agent, environment

    def unittest(
        self, environment=None, states=None, actions=None, num_episodes=None,
        experience_update=None, **agent
    ):
        """
        Generic unit-test.
        """
        if environment is None:
            environment = self.environment_spec(states=states, actions=actions)
            max_episode_timesteps = environment.pop('max_episode_timesteps')  # runner argument

        else:
            max_episode_timesteps = self.__class__.max_episode_timesteps

        agent = self.agent_spec(**agent)

        if num_episodes is None:
            num_updates = 2
        else:
            num_updates = None

        runner = Runner(
            agent=agent, environment=environment, max_episode_timesteps=max_episode_timesteps
        )
        runner.run(num_episodes=num_episodes, num_updates=num_updates, use_tqdm=False)
        runner.close()

        # Test experience-update, independent, deterministic
        if experience_update or (experience_update is None and self.__class__.experience_update):
            environment = Environment.create(
                environment=environment, max_episode_timesteps=max_episode_timesteps
            )
            agent = Agent.create(agent=agent, environment=environment)
            for episode in range(num_updates if num_episodes is None else num_episodes):
                episode_states = list()
                episode_internals = list()
                episode_actions = list()
                episode_terminal = list()
                episode_reward = list()
                states = environment.reset()
                internals = agent.initial_internals()
                terminal = False
                deterministic = True
                while not terminal:
                    episode_states.append(states)
                    episode_internals.append(internals)
                    actions, internals = agent.act(
                        states=states, internals=internals, independent=True,
                        deterministic=deterministic
                    )
                    deterministic = not deterministic
                    episode_actions.append(actions)
                    states, terminal, reward = environment.execute(actions=actions)
                    episode_terminal.append(terminal)
                    episode_reward.append(reward)
                agent.experience(
                    states=episode_states, internals=episode_internals, actions=episode_actions,
                    terminal=episode_terminal, reward=episode_reward
                )
                agent.update()

        self.finished_test()
