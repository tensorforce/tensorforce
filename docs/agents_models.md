Agent and model overview
========================

A reinforcement learning agent provides methods to process states and
return actions, to store past observations, and to load and save models.
Most agents employ a `Model` which implements the algorithms to
calculate the next action given the current state and to update model
parameters from past experiences.

> Environment <-> Runner <-> Agent <-> Model

Parameters to the agent are passed as a `Configuration` object. The
configuration is passed on to the `Model`.

Ready-to-use algorithms
-----------------------

We implemented some of the most common RL algorithms and try to keep
these up-to-date. Here we provide an overview over all implemented
agents and models.

### Agent / General parameters

`Agent` is the base class for all reinforcement learning agents. Every
agent inherits from this class.

```eval_rst
    .. autoclass:: tensorforce.agents.Agent
        :noindex:
        :show-inheritance:
        :members:
```

### Model

The `Model` class is the base class for reinforcement learning models.

```eval_rst
    .. autoclass:: tensorforce.models.Model
        :noindex:
        :show-inheritance:
        :members:
```


### MemoryAgent


```eval_rst
    .. autoclass:: tensorforce.agents.MemoryAgent
        :noindex:
        :show-inheritance:
        :members:
```


### BatchAgent


```eval_rst
    .. autoclass:: tensorforce.agents.BatchAgent
        :noindex:
        :show-inheritance:
        :members:
```


### Deep-Q-Networks (DQN)

```eval_rst
    .. autoclass:: tensorforce.agents.DQNAgent
        :noindex:
        :show-inheritance:
        :members:
```


### Normalized Advantage Functions


```eval_rst
    .. autoclass:: tensorforce.agents.NAFAgent
        :noindex:
        :show-inheritance:
        :members:
```

### Deep-Q-learning from demonstration (DQFD)

```eval_rst
    .. autoclass:: tensorforce.agents.DQFDAgent
        :noindex:
        :show-inheritance:
        :members:
```

### Vanilla Policy Gradient


```eval_rst
    .. autoclass:: tensorforce.agents.VPGAgent
        :noindex:
        :show-inheritance:
        :members:
```

### Trust Region Policy Optimization (TRPO)


```eval_rst
    .. autoclass:: tensorforce.agents.TRPOAgent
        :noindex:
        :show-inheritance:
        :members:
```

State preprocessing
-------------------

The agent handles state preprocessing. A preprocessor takes the raw state input
from the environment and modifies it (for instance, image resize, state 
concatenation, etc.). You can find information about our ready-to-use
preprocessors [here](preprocessing.html).


Building your own agent
-----------------------

If you want to build your own agent, it should always inherit from
`Agent`. If your agent uses a replay memory, it should probably inherit
from `MemoryAgent`, if it uses a batch replay that is emptied after each update,
it should probably inherit from `BatchAgent`.

We distinguish between agents and models. The `Agent` class handles the
interaction with the environment, such as state preprocessing, exploration
and observation of rewards. The `Model` class handles the mathematical
operations, such as building the tensorflow operations, calculating the
desired action and updating (i.e. optimizing) the model weights.

To start building your own agent, please refer to
[this blogpost](https://reinforce.io) to gain a deeper understanding of the
internals of the TensorForce library. Afterwards, have look on a sample
implementation, e.g. the [DQN Agent](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/agents/dqn_agent.py)
and [DQN Model](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/models/q_model.py).

