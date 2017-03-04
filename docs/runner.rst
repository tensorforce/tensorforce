Runners
=======

A "runner" is the code used to handle the interaction between the Environment and the Agent. TensorForce comes with ready-to-use runners. Of course, you can implement your own runners, too.

  Environment <-> Runner <-> Agent <-> Model

Ready-to-use runners
====================

We implemented a standard runner, a threaded runner (for asynchronous computations such as A3C) and a distributed runner if you would like to use multiple machines for computation.

Runner
------

This is the standard runner. It takes the following parameters for initialization:

.. code:: python

    from tensorforce.execution import Runner
    
    runner = Runner(
        agent = agent,  # tensorforce.agents.RLAgent object
        environment = env,  # tensorforce.environments.Environment object
        preprocessor = pp,  # state preprocessor, tensorforce.preprocessing.Preprocessor or *.Stack object
        repeat_actions = int,  # how often to repeat actions
    )
    
A reinforcement learning agent observes states from the environment, selects actions and collect experience which is used to update its model and improve action selection. You can get information about our ready-to-use agents `here <agents_models.rst>`__.

The environment object is either the "real" environment, or a proxy which fulfills the actions selected by the agent in the real world. You can find information about environments `here <environments.rst>`__.

A preprocessor takes the raw state input from the environment and modifies it (for instance, image resize, state concatenation, etc.). You can find information about our ready-to-use preprocessors `here <preprocessing.rst>`__.
    
The repeat_action parameter indicates how often to repeat an action in the environment before passing the next state to the agent. Rewards during action repeat are cumulated, and the terminal status is preserved. Please note that some environments, such as the Atari environments from `OpenAI Gym <https://gym.openai.com/>`__, automatically repeat actions for a variable amount of states (in this case, for 3-5 frames).

The runner is started with the ``Runner.run(...)`` method:

.. code:: python

    runner.run(
        episodes = int,  # number of episodes to run
        max_timesteps = int,  # maximum timesteps per episode
        episode_finished = object,  # callback function called when episode is finished
    )
    
You can use the episode_finished callback for printing performance feedback:

.. code:: python

    def episode_finished(r):
        if r.episode % 10 == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
        return True

Using the Runner
~~~~~~~~~~~~~~~~

Here is some example code for using the runner (without preprocessing).

.. code:: python

    from tensorforce.config import Config
    from tensorforce.external.openai_gym import OpenAIGymEnvironment
    from tensorforce.execution import Runner
    from tensorforce.examples.simple_q_agent import SimpleQAgent

    def main():
        gym_id = 'CartPole-v0'
        max_episodes = 10000
        max_timesteps = 1000
    
        env = OpenAIGymEnvironment(gym_id, monitor=False, monitor_video=False)
    
        config = Config({
            'repeat_actions': 1,
            'actions': env.actions,
            'action_shape': env.action_shape,
            'state_shape': env.state_shape,
            'exploration': 'constant',
            'exploration_args': [0.1]
        })
    
        agent = SimpleQAgent(config, "simpleq")
    
        runner = Runner(agent, env)
    
        def episode_finished(r):
            if r.episode % 10 == 0:
                print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
                print("Episode reward: {}".format(r.episode_rewards[-1]))
                print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
            return True
    
        print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))
        runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
        print("Learning finished. Total episodes: {ep}".format(ep=runner.episode + 1))
    
    if __name__ == '__main__':
        main()


ThreadRunner
------------
No description, yet.

DistributedRunner
-----------------
No description, yet.

Building your own runner
========================

There are three mandatory tasks any runner implements: Getting the action from the agent, passing it to the environment, and passing the resulting observation to the agent.

.. code:: python

    # Get action
    action = agent.get_action(state, self.episode)
    
    # Execute action in the environment
    result = environment.execute_action(action)

    # Pass observation to the agent
    agent.add_observation(state, action, result['reward'], result['terminal_state'])
    
There are other tasks a runner could implement, such as `preprocessing <preprocessing.rst>`__, repeating actions and storing episode rewards.

If you would like to build your own runner, it is probably a good idea to take a look at the `source code of our Runner class <../tensorforce/execution/runner.py>`__.

