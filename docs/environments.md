Environments
============

A reinforcement learning environment provides the API to a simulated or real
environment as the subject for optimization. It could be anything from
video games (e.g. Atari) to robots or trading systems. The agent interacts
with this environment and learns to act optimally in its dynamics.

> Environment <-> Runner <-> Agent <-> Model

```eval_rst
    .. autoclass:: tensorforce.environments.Environment
        :members:
```


Ready-to-use environments
-------------------------

### OpenAI Gym

```eval_rst
    .. autoclass:: tensorforce.environments.openai_gym.OpenAIGym
        :show-inheritance:
        :members:
        :special-members: __init__
```

### OpenAI Universe

```eval_rst
    .. autoclass:: tensorforce.environments.openai_universe.OpenAIUniverse
        :show-inheritance:
        :members:
        :special-members: __init__
```

### Deepmind Lab

```eval_rst
    .. autoclass:: tensorforce.environments.deepmind_lab.DeepMindLab
        :show-inheritance:
        :members:
        :special-members: __init__
```
