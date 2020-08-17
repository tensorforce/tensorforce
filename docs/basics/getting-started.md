Getting started
===============


[Quickstart example](https://github.com/tensorforce/tensorforce/blob/master/examples/quickstart.py)


### Initializing an environment

It is recommended to initialize an environment via the `Environment.create(...)` [interface](../environments/environment.html).

```python
from tensorforce.environments import Environment
```

For instance, the [OpenAI CartPole environment](../environments/openai_gym.html) can be initialized as follows (see environment docs for available environments and arguments):

```python
environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)
```

Gym's pre-defined versions are also accessible:

```python
environment = Environment.create(environment='gym', level='CartPole-v1')
```

Alternatively, an environment can be specified as a config file:

```json
{
    "environment": "gym",
    "level": "CartPole"
}
```

Environment config files can be loaded by passing their file path:

```python
environment = Environment.create(
    environment='environment.json', max_episode_timesteps=500
)
```

Custom Gym environments can be used in the same way, but require the corresponding class(es) to be imported and registered accordingly.

Finally, it is possible to implement a custom environment using Tensorforce's `Environment` interface:

```python
class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='float', shape=(8,))

    def actions(self):
        return dict(type='int', num_values=4)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; restrict training timesteps via
    #     Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = np.random.random() < 0.5
        reward = np.random.random()
        return next_state, terminal, reward
```

Custom environment implementations can be loaded by passing either the environment object itself:

```python
environment = Environment.create(
    environment=CustomEnvironment, max_episode_timesteps=100
)
```

or its module path:

```python
environment = Environment.create(
    environment='custom_env.CustomEnvironment', max_episode_timesteps=100
)
```

It is generally recommended to specify the `max_episode_timesteps` argument of `Environment.create(...)` (at least for training), as some agent parameters may rely on this value.




### Initializing an agent

Similarly to environments, it is recommended to initialize an agent via the `Agent.create(...)` [interface](../agents/agent.html).

```python
from tensorforce.agents import Agent
```

For instance, the [generic Tensorforce agent](../agents/tensorforce.html) can be initialized as follows (see agent docs for available agents and arguments):

```python
agent = Agent.create(
    agent='tensorforce', environment=environment, update=64,
    objective='policy_gradient', reward_estimation=dict(horizon=20)
)
```

Other pre-defined agent classes can alternatively be used, for instance, [Proximal Policy Optimization](../agents/ppo.html):

```python
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)
```

Alternatively, an agent can be specified as a config file:

```json
{
    "agent": "tensorforce",
    "update": 64,
    "objective": "policy_gradient",
    "reward_estimation": {
        "horizon": 20
    }
}
```

Agent config files can be loaded by passing their file path:

```python
agent = Agent.create(agent='agent.json', environment=environment)
```

While it is possible to specify the agent arguments `states`, `actions` and `max_episode_timesteps`, it is generally recommended to specify the `environment` argument instead (which will automatically infer the other values accordingly), by passing the environment object as returned by `Environment.create(...)`.




### Training and evaluation

It is recommended to use the execution utilities for training and evaluation, like the [Runner utility](../execution/runner.html), which offer a range of configuration options:

```python
from tensorforce.execution import Runner
```

A basic experiment consisting of training and subsequent evaluation can be written in a few lines of code:

```python
runner = Runner(
    agent='agent.json',
    environment=dict(environment='gym', level='CartPole'),
    max_episode_timesteps=500
)

runner.run(num_episodes=200)

runner.run(num_episodes=100, evaluation=True)

runner.close()
```

The same interface also makes it possible to run experiments involving multiple parallelized environments:

```python
runner = Runner(
    agent='agent.json',
    environment=dict(environment='gym', level='CartPole'),
    max_episode_timesteps=500,
    num_parallel=5, remote='multiprocessing'
)

runner.run(num_episodes=100)

runner.close()
```

Note that in this case both agent and environment are created as part of `Runner`, not via `Agent.create(...)` and `Environment.create(...)`. If agent and environment are specified separately, the user is required to take care of passing the agent arguments `environment` and `parallel_interactions` (in the parallelized case) as well as closing both agent and environment separately at the end.

The execution utility classes take care of handling the agent-environment interaction correctly, and thus should be used where possible. Alternatively, if more detailed control over the agent-environment interaction is required, a simple training and evaluation loop can be defined as follows:

```python
# Create agent and environment
environment = Environment.create(
    environment='environment.json', max_episode_timesteps=500
)
agent = Agent.create(agent='agent.json', environment=environment)

# Train for 200 episodes
for _ in range(200):
    states = environment.reset()
    terminal = False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

# Evaluate for 100 episodes
sum_rewards = 0.0
for _ in range(100):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals,
            deterministic=True, independent=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward

print('Mean episode reward:', sum_rewards / 100)

# Close agent and environment
agent.close()
environment.close()
```
