Getting started
===============


##### Training

```python
from tensorforce.agents import Agent
from tensorforce.environments import Environment

# Setup environment
# (Tensorforce or custom implementation, ideally using the Environment interface)
environment = Environment.create(environment='environment.json')

# Create and initialize agent
agent = Agent.create(agent='agent.json', environment=environment)
agent.initialize()

# Reset agent and environment at the beginning of a new episode
agent.reset()
states = environment.reset()
terminal = False

# Agent-environment interaction training loop
while not terminal:
    actions = agent.act(states=states)
    states, terminal, reward = environment.execute(actions=actions)
    agent.observe(terminal=terminal, reward=reward)

# Close agent and environment
agent.close()
environment.close()
```


##### Evaluation / application

```python
# Agent-environment interaction evaluation loop
while not terminal:
    actions = agent.act(states=states, evaluation=True)
    states, terminal, reward = environment.execute(actions=actions)
```


##### Runner utility

```python
from tensorforce.execution import Runner

# Tensorforce runner utility
runner = Runner(agent='agent.json', environment='environment.json')

# Run training
runner.run(num_episodes=500)

# Close runner
runner.close()
```
