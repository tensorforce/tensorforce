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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY K , either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
from tqdm import tqdm

import numpy as np

from tensorforce import util
from tensorforce.agents import Agent
from tensorforce.environments import Environment


class Runner(object):
    """
    Tensorforce runner utility.

    Args:
        agent (specification | Agent object): Agent specification or object, the latter is not
            closed automatically as part of `runner.close()`
            (<span style="color:#C00000"><b>required</b></span>).
        environment (specification | Environment object): Environment specification or object, the
            latter is not closed automatically as part of `runner.close()`
            (<span style="color:#C00000"><b>required</b></span>).
        max_episode_timesteps (int > 0): Maximum number of timesteps per episode, overwrites the
            environment default if defined
            (<span style="color:#00C000"><b>default</b></span>: environment default).
        evaluation_environment (specification | Environment object): Evaluation environment or
            object, the latter is not closed automatically as part of `runner.close()`
            (<span style="color:#00C000"><b>default</b></span>: none).
        save_best_agent (string): Directory to save the best version of the agent according to the
            evaluation
            (<span style="color:#00C000"><b>default</b></span>: best agent is not saved).
    """

    def __init__(
        self, agent, environment, max_episode_timesteps=None, evaluation_environment=None,
        save_best_agent=None
    ):
        self.is_environment_external = isinstance(environment, Environment)
        self.environment = Environment.create(
            environment=environment, max_episode_timesteps=max_episode_timesteps
        )

        if evaluation_environment is None:
            self.evaluation_environment = None
        else:
            self.is_eval_environment_external = isinstance(evaluation_environment, Environment)
            self.evaluation_environment = Environment.create(
                environment=evaluation_environment, max_episode_timesteps=max_episode_timesteps
            )
            assert self.evaluation_environment.states() == self.environment.states()
            assert self.evaluation_environment.actions() == self.environment.actions()

        self.is_agent_external = isinstance(agent, Agent)
        self.agent = Agent.create(agent=agent, environment=self.environment)
        self.save_best_agent = save_best_agent

        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.episode_seconds = list()
        self.episode_agent_seconds = list()

    def close(self):
        if hasattr(self, 'tqdm'):
            self.tqdm.close()
        if not self.is_agent_external:
            self.agent.close()
        if not self.is_environment_external:
            self.environment.close()
        if self.evaluation_environment is not None and not self.is_eval_environment_external:
            self.evaluation_environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(
        self,
        # General
        num_episodes=None, num_timesteps=None, num_updates=None, num_repeat_actions=1,
        # Callback
        callback=None, callback_episode_frequency=None, callback_timestep_frequency=None,
        # Tqdm
        use_tqdm=True, mean_horizon=1,
        # Evaluation
        evaluation=False, evaluation_callback=None, evaluation_frequency=None,
        num_evaluation_iterations=1
    ):
        # General
        if num_episodes is None:
            self.num_episodes = float('inf')
        else:
            self.num_episodes = num_episodes
        if num_timesteps is None:
            self.num_timesteps = float('inf')
        else:
            self.num_timesteps = num_timesteps
        if num_updates is None:
            self.num_updates = float('inf')
        else:
            self.num_updates = num_updates
        self.num_repeat_actions = num_repeat_actions

        # Callback
        assert callback_episode_frequency is None or callback_timestep_frequency is None
        if callback_episode_frequency is None and callback_timestep_frequency is None:
            callback_episode_frequency = 1
        if callback_episode_frequency is None:
            self.callback_episode_frequency = float('inf')
        else:
            self.callback_episode_frequency = callback_episode_frequency
        if callback_timestep_frequency is None:
            self.callback_timestep_frequency = float('inf')
        else:
            self.callback_timestep_frequency = callback_timestep_frequency
        if callback is None:
            self.callback = (lambda r: True)
        elif util.is_iterable(x=callback):
            def sequential_callback(runner):
                result = True
                for fn in callback:
                    x = fn(runner)
                    if isinstance(result, bool):
                        result = result and x
                return result
            self.callback = sequential_callback
        else:
            def boolean_callback(runner):
                result = callback(runner)
                if isinstance(result, bool):
                    return result
                else:
                    return True
            self.callback = boolean_callback

        # Timestep/episode/update counter
        self.timesteps = 0
        self.episodes = 0
        self.updates = 0

        # Tqdm
        if use_tqdm:
            if hasattr(self, 'tqdm'):
                self.tqdm.close()

            assert self.num_episodes != float('inf') or self.num_timesteps != float('inf')
            inner_callback = self.callback

            if self.num_episodes != float('inf'):
                # Episode-based tqdm (default option if both num_episodes and num_timesteps set)
                assert self.num_episodes != float('inf')
                bar_format = (
                    '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, reward={postfix[0]:.2f}, ts/ep='
                    '{postfix[1]}, sec/ep={postfix[2]:.2f}, ms/ts={postfix[3]:.1f}, agent='
                    '{postfix[4]:.1f}%]'
                )
                postfix = [0.0, 0, 0.0, 0.0, 0.0]
                self.tqdm = tqdm(
                    desc='Episodes', total=self.num_episodes, bar_format=bar_format,
                    initial=self.episodes, postfix=postfix
                )
                self.tqdm_last_update = self.episodes

                def tqdm_callback(runner):
                    mean_reward = float(np.mean(runner.episode_rewards[-mean_horizon:]))
                    mean_ts_per_ep = int(np.mean(runner.episode_timesteps[-mean_horizon:]))
                    mean_sec_per_ep = float(np.mean(runner.episode_seconds[-mean_horizon:]))
                    mean_agent_sec = float(np.mean(runner.episode_agent_seconds[-mean_horizon:]))
                    mean_ms_per_ts = mean_sec_per_ep * 1000.0 / mean_ts_per_ep
                    mean_rel_agent = mean_agent_sec * 100.0 / mean_sec_per_ep
                    runner.tqdm.postfix[0] = mean_reward
                    runner.tqdm.postfix[1] = mean_ts_per_ep
                    runner.tqdm.postfix[2] = mean_sec_per_ep
                    runner.tqdm.postfix[3] = mean_ms_per_ts
                    runner.tqdm.postfix[4] = mean_rel_agent
                    runner.tqdm.update(n=(runner.episodes - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.episodes
                    return inner_callback(runner)

            else:
                # Timestep-based tqdm
                assert self.num_timesteps != float('inf')
                self.tqdm = tqdm(
                    desc='Timesteps', total=self.num_timesteps, initial=self.timesteps,
                    postfix=dict(mean_reward='n/a')
                )
                self.tqdm_last_update = self.timesteps

                def tqdm_callback(runner):
                    # sum_timesteps_reward = sum(runner.timestep_rewards[num_mean_reward:])
                    # num_timesteps = min(num_mean_reward, runner.episode_timestep)
                    # mean_reward = sum_timesteps_reward / num_episodes
                    runner.tqdm.set_postfix(mean_reward='n/a')
                    runner.tqdm.update(n=(runner.timesteps - runner.tqdm_last_update))
                    runner.tqdm_last_update = runner.timesteps
                    return inner_callback(runner)

            self.callback = tqdm_callback

        # Evaluation
        self.evaluation = evaluation
        if evaluation_callback is None:
            self.evaluation_callback = (lambda r: None)
        else:
            assert not self.evaluation
            self.evaluation_callback = evaluation_callback
        if self.evaluation:
            assert evaluation_frequency is None
        self.evaluation_frequency = evaluation_frequency
        self.num_evaluation_iterations = num_evaluation_iterations
        if self.save_best_agent is not None:
            assert not self.evaluation
            inner_evaluation_callback = self.evaluation_callback

            def mean_reward_callback(runner):
                result = inner_evaluation_callback(runner)
                if result is None:
                    return float(np.mean(runner.evaluation_rewards))
                else:
                    return result

            self.evaluation_callback = mean_reward_callback
            self.best_evaluation_score = None

        # Required if agent was previously stopped mid-episode
        self.agent.reset()

        # Episode loop
        while True:
            # Run episode
            if not self.run_episode(environment=self.environment, evaluation=self.evaluation):
                return

            # Increment episode counter (after calling callback)
            self.episodes += 1

            # Update experiment statistics
            self.episode_rewards.append(self.episode_reward)
            self.episode_timesteps.append(self.episode_timestep)
            self.episode_seconds.append(self.episode_second)
            self.episode_agent_seconds.append(self.episode_agent_second)

            # Run evaluation
            if self.evaluation_frequency is None:
                is_evaluation = False
            elif self.evaluation_frequency == 'update':
                is_evaluation = self.episode_updated
            else:
                is_evaluation = (self.episodes % self.evaluation_frequency == 0)
            if is_evaluation:
                if self.evaluation_environment is None:
                    environment = self.environment
                else:
                    environment = self.evaluation_environment

                self.evaluation_rewards = list()
                self.evaluation_timesteps = list()
                self.evaluation_seconds = list()
                self.evaluation_agent_seconds = list()

                # Evaluation loop
                for _ in range(self.num_evaluation_iterations):
                    self.run_episode(environment=environment, evaluation=True)

                    self.evaluation_rewards.append(self.episode_reward)
                    self.evaluation_timesteps.append(self.episode_timestep)
                    self.evaluation_seconds.append(self.episode_second)
                    self.evaluation_agent_seconds.append(self.episode_agent_second)

                # Evaluation callback
                if self.save_best_agent is not None:
                    evaluation_score = self.evaluation_callback(self)
                    assert isinstance(evaluation_score, float)
                    if self.best_evaluation_score is None:
                        self.best_evaluation_score = evaluation_score
                    elif evaluation_score > self.best_evaluation_score:
                        self.best_evaluation_score = evaluation_score
                        self.agent.save(
                            directory=self.save_best_agent, filename='best-model',
                            append_timestep=False
                        )
                else:
                    self.evaluation_callback(self)

            # # Update global timestep/episode/update
            # self.global_timesteps = self.agent.timesteps
            # self.global_episodes = self.agent.episodes
            # self.global_updates = self.agent.updates

            # Callback
            if self.episodes % self.callback_episode_frequency == 0 and not self.callback(self):
                return

            # Terminate experiment if too long
            if self.timesteps >= self.num_timesteps:
                return
            # elif self.evaluation and self.timesteps >= self.num_timesteps:
            #     return
            elif self.episodes >= self.num_episodes:
                return
            # elif self.evaluation and self.episodes >= self.num_episodes:
            #     return
            elif self.updates >= self.num_updates:
                return
            # elif self.evaluation and self.updates >= self.num_updates:
            #     return
            elif self.agent.should_stop():
                return

    def run_episode(self, environment, evaluation):
        # Episode statistics
        self.episode_reward = 0
        self.episode_timestep = 0
        self.episode_updated = False
        self.episode_agent_second = 0.0
        episode_start = time.time()

        # Start environment episode
        states = environment.reset()

        # Timestep loop
        while True:
            # Retrieve actions from agent
            agent_start = time.time()
            actions = self.agent.act(states=states, evaluation=evaluation)
            self.timesteps += 1
            self.episode_agent_second += time.time() - agent_start
            self.episode_timestep += 1
            # Execute actions in environment (optional repeated execution)
            reward = 0.0
            for _ in range(self.num_repeat_actions):
                states, terminal, step_reward = environment.execute(actions=actions)
                if isinstance(terminal, bool):
                    terminal = int(terminal)
                reward += step_reward
                if terminal > 0:
                    break
            self.episode_reward += reward

            # Observe unless evaluation
            if not evaluation:
                agent_start = time.time()
                updated = self.agent.observe(terminal=terminal, reward=reward)
                self.updates += int(updated)
                self.episode_agent_second += time.time() - agent_start
                self.episode_updated = self.episode_updated or updated

            # Episode termination check
            if terminal > 0:
                break

            # No callbacks for evaluation
            if evaluation:
                continue

            # Callback
            if self.episode_timestep % self.callback_timestep_frequency == 0 and \
                    not self.callback(self):
                return False

            # # Update global timestep/episode/update
            # self.global_timesteps = self.agent.timesteps
            # self.global_episodes = self.agent.episodes
            # self.global_updates = self.agent.updates

            # Terminate experiment if too long
            if self.timesteps >= self.num_timesteps:
                return False
            elif self.episodes >= self.num_episodes:
                return False
            elif self.updates >= self.num_updates:
                return False
            elif self.agent.should_stop():
                return False

        # Update episode statistics
        self.episode_second = time.time() - episode_start

        return True
