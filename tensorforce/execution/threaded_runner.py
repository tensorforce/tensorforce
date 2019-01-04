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

import importlib
from inspect import getargspec
import threading
import time
import warnings

from tensorforce import TensorforceError
from tensorforce.execution.base_runner import BaseRunner
from tensorforce.agents import DRLAgent
from tensorforce.agents import agents as AgentsDictionary


class ThreadedRunner(BaseRunner):
    """
    Runner for non-realtime threaded execution of multiple agents.
    """

    def __init__(self, agent, environment, repeat_actions=1, save_path=None, save_episodes=None, save_frequency=None,
                 save_frequency_unit=None, agents=None, environments=None):
        """
        Initialize a ThreadedRunner object.

        Args:
            save_path (str): Path where to save the shared model.
            save_episodes (int): Deprecated: Every how many (global) episodes do we save the shared model?
            save_frequency (int): The frequency with which to save the model (could be sec, steps, or episodes).
            save_frequency_unit (str): "s" (sec), "t" (timesteps), "e" (episodes)
            agents (List[Agent]): Deprecated: List of Agent objects. Use `agent`, instead.
            environments (List[Environment]): Deprecated: List of Environment objects. Use `environment`, instead.
        """
        if agents is not None:
            warnings.warn("WARNING: `agents` parameter is deprecated, use `agent` instead.",
                          category=DeprecationWarning)
            agent = agents
        if environments is not None:
            warnings.warn("WARNING: `environments` parameter is deprecated, use `environments` instead.",
                          category=DeprecationWarning)
            environment = environments
        super(ThreadedRunner, self).__init__(agent, environment, repeat_actions)

        if len(agent) != len(environment):
            raise TensorforceError("Each agent must have its own environment. Got {a} agents and {e} environments.".
                                   format(a=len(self.agent), e=len(self.environment)))
        self.save_path = save_path
        self.save_episodes = save_episodes
        if self.save_episodes is not None:
            warnings.warn("WARNING: `save_episodes` parameter is deprecated, use `save_frequency` AND "
                          "`save_frequency_unit` instead.",
                          category=DeprecationWarning)
            self.save_frequency = self.save_episodes
            self.save_frequency_unit = "e"
        else:
            self.save_frequency = save_frequency
            self.save_frequency_unit = save_frequency_unit

        # Initialize stats for parallel runs.
        self.episode_list_lock = threading.Lock()
        # Stop-condition flag that each worker abides to (aborts if True).
        self.should_stop = False
        # Global time counter (sec).
        self.time = None

    def close(self):
        self.agent[0].close()  # only close first agent as we just have one shared model
        for e in self.environment:
            e.close()

    def run(
        self,
        num_episodes=-1,
        max_episode_timesteps=-1,
        episode_finished=None,
        summary_report=None,
        summary_interval=0,
        num_timesteps=None,
        deterministic=False,
        episodes=None,
        max_timesteps=None,
        testing=False,
        sleep=None
    ):
        """
        Executes this runner by starting all Agents in parallel (each one in one thread).

        Args:
            episodes (int): Deprecated; see num_episodes.
            max_timesteps (int): Deprecated; see max_episode_timesteps.
        """

        # Renamed episodes into num_episodes to match BaseRunner's signature (fully backw. compatible).
        if episodes is not None:
            num_episodes = episodes
            warnings.warn("WARNING: `episodes` parameter is deprecated, use `num_episodes` instead.",
                          category=DeprecationWarning)
        assert isinstance(num_episodes, int)
        # Renamed max_timesteps into max_episode_timesteps to match single Runner's signature (fully backw. compatible).
        if max_timesteps is not None:
            max_episode_timesteps = max_timesteps
            warnings.warn("WARNING: `max_timesteps` parameter is deprecated, use `max_episode_timesteps` instead.",
                          category=DeprecationWarning)
        assert isinstance(max_episode_timesteps, int)

        if summary_report is not None:
            warnings.warn("WARNING: `summary_report` parameter is deprecated, use `episode_finished` callback "
                          "instead to generate summaries every n episodes.",
                          category=DeprecationWarning)

        self.reset()

        # Reset counts/stop-condition for this run.
        self.global_episode = 0
        self.global_timestep = 0
        self.should_stop = False

        # Create threads.
        threads = [threading.Thread(target=self._run_single, args=(t, self.agent[t], self.environment[t],),
                                    kwargs={"deterministic": deterministic,
                                            "max_episode_timesteps": max_episode_timesteps,
                                            "episode_finished": episode_finished,
                                            "testing": testing,
                                            "sleep": sleep})
                   for t in range(len(self.agent))]

        # Start threads.
        self.start_time = time.time()
        [t.start() for t in threads]

        # Stay idle until killed by SIGINT or a global stop condition is met.
        try:
            next_summary = 0
            next_save = 0 if self.save_frequency_unit != "s" else time.time()
            while any([t.is_alive() for t in threads]) and self.global_episode < num_episodes or num_episodes == -1:
                self.time = time.time()

                # This is deprecated (but still supported) and should be covered by the `episode_finished` callable.
                if summary_report is not None and self.global_episode > next_summary:
                    summary_report(self)
                    next_summary += summary_interval

                if self.save_path and self.save_frequency is not None:
                    do_save = True
                    current = None
                    if self.save_frequency_unit == "e" and self.global_episode > next_save:
                        current = self.global_episode
                    elif self.save_frequency_unit == "s" and self.time > next_save:
                        current = self.time
                    elif self.save_frequency_unit == "t" and self.global_timestep > next_save:
                        current = self.global_timestep
                    else:
                        do_save = False

                    if do_save:
                        self.agent[0].save_model(self.save_path)
                        # Make sure next save is later than right now.
                        while next_save < current:
                            next_save += self.save_frequency
                time.sleep(1)

        except KeyboardInterrupt:
            print('Keyboard interrupt, sending stop command to threads')

        self.should_stop = True

        # Join threads.
        [t.join() for t in threads]
        print('All threads stopped')

    def _run_single(self, thread_id, agent, environment, deterministic=False,
                    max_episode_timesteps=-1, episode_finished=None, testing=False, sleep=None):
        """
        The target function for a thread, runs an agent and environment until signaled to stop.
        Adds rewards to shared episode rewards list.

        Args:
            thread_id (int): The ID of the thread that's running this target function.
            agent (Agent): The Agent object that this particular thread uses.
            environment (Environment): The Environment object that this particular thread uses.
            max_episode_timesteps (int): Max. number of timesteps per episode. Use -1 or 0 for non-limited episodes.
            episode_finished (callable): Function called after each episode that takes an episode summary spec and
                returns False, if this single run should terminate after this episode.
                Can be used e.g. to set a particular mean reward threshold.
        """

        # figure out whether we are using the deprecated way of "episode_finished" reporting
        old_episode_finished = False
        if episode_finished is not None and len(getargspec(episode_finished).args) == 1:
            old_episode_finished = True

        episode = 0
        # Run this single worker (episode loop) as long as global count thresholds have not been reached.
        while not self.should_stop:
            state = environment.reset()
            agent.reset()
            self.global_timestep, self.global_episode = agent.timestep, agent.episode
            episode_reward = 0

            # Time step (within episode) loop
            time_step = 0
            time_start = time.time()
            while True:
                action, internals, states = agent.act(states=state, deterministic=deterministic, buffered=False)
                reward = 0
                for repeat in range(self.repeat_actions):
                    state, terminal, step_reward = environment.execute(action=action)
                    reward += step_reward
                    if terminal:
                        break

                if not testing:
                    # agent.observe(reward=reward, terminal=terminal)
                    # Insert everything at once.
                    agent.atomic_observe(
                        states=state,
                        actions=action,
                        internals=internals,
                        reward=reward,
                        terminal=terminal
                    )

                if sleep is not None:
                    time.sleep(sleep)

                time_step += 1
                episode_reward += reward

                if terminal or time_step == max_episode_timesteps:
                    break

                # Abort the episode (discard its results) when global says so.
                if self.should_stop:
                    return

            self.global_timestep += time_step

            # Avoid race condition where order in episode_rewards won't match order in episode_timesteps.
            self.episode_list_lock.acquire()
            self.episode_rewards.append(episode_reward)
            self.episode_timesteps.append(time_step)
            self.episode_times.append(time.time() - time_start)
            self.episode_list_lock.release()

            if episode_finished is not None:
                # old way of calling episode_finished
                if old_episode_finished:
                    summary_data = {
                        "thread_id": thread_id,
                        "episode": episode,
                        "timestep": time_step,
                        "episode_reward": episode_reward
                    }
                    if not episode_finished(summary_data):
                        return
                # New way with BasicRunner (self) and thread-id.
                elif not episode_finished(self, thread_id):
                    return

            episode += 1

    # Backwards compatibility for deprecated properties (in case someone directly references these).
    @property
    def agents(self):
        return self.agent

    @property
    def environments(self):
        return self.environment

    @property
    def episode_lengths(self):
        return self.episode_timesteps

    @property
    def global_step(self):
        return self.global_timestep


def WorkerAgentGenerator(agent_class):
    """
    Worker Agent generator, receives an Agent class and creates a Worker Agent class that inherits from that Agent.
    """

    # Support special case where class is given as type-string (AgentsDictionary) or class-name-string.
    if isinstance(agent_class, str):
        agent_class = AgentsDictionary.get(agent_class)
        # Last resort: Class name given as string?
        if not agent_class and agent_class.find('.') != -1:
            module_name, function_name = agent_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            agent_class = getattr(module, function_name)

    class WorkerAgent(agent_class):
        """
        Worker agent receiving a shared model to avoid creating multiple models.
        """

        def __init__(self, model=None, **kwargs):
            # Set our model externally.
            self.model = model
            # Be robust against `network` coming in from kwargs even though this agent doesn't have one
            if not issubclass(agent_class, DRLAgent):
                kwargs.pop("network")
            # Call super c'tor (which will call initialize_model and assign self.model to the return value).
            super(WorkerAgent, self).__init__(**kwargs)

        def initialize_model(self):
            # Return our model (already given and initialized).
            return self.model

    return WorkerAgent


def clone_worker_agent(agent, factor, environment, network, agent_config):
    """
    Clones a given Agent (`factor` times) and returns a list of the cloned Agents with the original Agent
    in the first slot.

    Args:
        agent (Agent): The Agent object to clone.
        factor (int): The length of the final list.
        environment (Environment): The Environment to use for all cloned agents.
        network (LayeredNetwork): The Network to use (or None) for an Agent's Model.
        agent_config (dict): A dict of Agent specifications passed into the Agent's c'tor as kwargs.
    Returns:
        The list with `factor` cloned agents (including the original one).
    """
    ret = [agent]
    for i in range(factor - 1):
        worker = WorkerAgentGenerator(type(agent))(
            states=environment.states,
            actions=environment.actions,
            network=network,
            model=agent.model,
            **agent_config
        )
        ret.append(worker)

    return ret
