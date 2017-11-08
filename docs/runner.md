Runners
=======

A "runner" manages the interaction between the Environment and the
Agent. TensorForce comes with ready-to-use runners. Of course, you can
implement your own runners, too. If you are not using simulation
environments, the runner is simply your application code using the Agent
API.

> Environment <-> Runner <-> Agent <-> Model

Ready-to-use runners
--------------------

We implemented a standard runner, a threaded runner (for real-time
interaction e.g. with OpenAI Universe) and a distributed runner for A3C
variants.

### Runner

This is the standard runner. It requires an agent and an environment for
initialization:

```python
from tensorforce.execution import Runner

runner = Runner(
    agent = agent,  # Agent object
    environment = env  # Environment object
)
```

A reinforcement learning agent observes states from the environment,
selects actions and collect experience which is used to update its model
and improve action selection. You can get information about our
ready-to-use agents [here](agents_models.html).

The environment object is either the "real" environment, or a proxy
which fulfills the actions selected by the agent in the real world. You
can find information about environments [here](environments.html).

The runner is started with the `Runner.run(...)` method:

```python
runner.run(
    episodes = int,  # number of episodes to run
    max_timesteps = int,  # maximum timesteps per episode
    episode_finished = object,  # callback function called when episode is finished
)
runner.close()
```

You can use the episode\_finished callback for printing performance
feedback:

```python
def episode_finished(r):
    if r.episode % 10 == 0:
        print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
        print("Episode reward: {}".format(r.episode_rewards[-1]))
        print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
    return True
```

#### Using the Runner

Here is some example code for using the runner (without preprocessing).

```python
import logging

from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner

def main():
    gym_id = 'CartPole-v0'
    max_episodes = 10000
    max_timesteps = 1000

    env = OpenAIGym(gym_id)
    network_spec = [
        dict(type='dense', size=32, activation='tanh'),
        dict(type='dense', size=32, activation='tanh')
    ]

    agent = DQNAgent(
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network_spec,
        batch_size=64
    )

    runner = Runner(agent, env)
    
    report_episodes = 10

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            logging.info("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            logging.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logging.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
    runner.close()

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()
```


Building your own runner
------------------------

There are three mandatory tasks any runner implements: Obtaining an
action from the agent, passing it to the environment, and passing the
resulting observation to the agent.

```python
# Get action
action = agent.act(state)

# Execute action in the environment
state, reward, terminal_state = environment.execute(action)

# Pass observation to the agent
agent.observe(state, action, reward, terminal_state)
```

The key idea here is the separation of concerns. External code should
not need to manage batches or remember network features, this is that
the agent is for. Conversely, an agent need not concern itself with how
a model is implemented and the API should facilitate easy combination of
different agents and models.

If you would like to build your own runner, it is probably a good idea
to take a look at the [source code of our Runner
class](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/execution/runner.py).
