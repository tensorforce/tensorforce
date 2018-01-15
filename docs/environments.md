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
        :noindex:
```


Ready-to-use environments
-------------------------

### OpenAI Gym

```eval_rst
    .. autoclass:: tensorforce.contrib.openai_gym.OpenAIGym
        :noindex:
        :show-inheritance:
        :members:
        :special-members: __init__
```

### OpenAI Universe

```eval_rst
    .. autoclass:: tensorforce.contrib.openai_universe.OpenAIUniverse
        :noindex:
        :show-inheritance:
        :members:
        :special-members: __init__
```

### Deepmind Lab

```eval_rst
    .. autoclass:: tensorforce.contrib.deepmind_lab.DeepMindLab
        :noindex:
        :show-inheritance:
        :members:
        :special-members: __init__
```

### Unreal Engine 4 Games

```eval_rst
    .. autoclass:: tensorforce.contrib.unreal_engine.UE4Environment
        :noindex:
        :show-inheritance:
        :members:
        :special-members: __init__
```
