# Agent overview

## <a name="RLAgent">RLAgent / General parameters</a>

```RLAgent``` is the base class for all reinforcement learning agents. Every agent inherits from this class.

### Parameters

```python
# General parameters
state_shape: tuple  # shape of state tensor
action_shape: tuple  # shape of action tensor
state_type: type  # type of state tensor (default: np.float32)
action_type: type  # type of action tensor (default: np.int)
reward_type: type  # type of reward signal (default: np.float32)
deterministic_mode: boolean  # Use a deterministic random function (default: False)
```

## <a name="MemoryAgent">MemoryAgent</a>

The ```MemoryAgent``` class implements a replay memory, from which it samples batches to update the value function.

### Parameters:

```python
# Parameters for MemoryAgent
memory_capacity: int  # maxmium capacity for replay memory
batch_size: int  # batch size for updates
update_rate: float  # how often to update the value function (e.g. 0.25 means every 4th step)
use_target_network: boolean  # does the model use a target network?
target_network_update_rate: float  # how often to update the target network
min_replay_size: int  # minimum replay memory size to reach before first update
update_repeat: int  # how often to repeat updates
```

## <a name="DQNAgent">DQNAgent</a>

Standard DQN agent.

### Parameters:
Uses all [general parameters](#RLAgent) and all parameters from the [MemoryAgent](#MemoryAgent). Additional parameters:

```python
# Parameters for DQNModel
double_dqn: boolean  # use double dqn network or not
tau: float  # indicate how fast the target network should be updated
gamma: float  # discount factor
alpha: float  # learning rate
clip_gradients: boolean  # clip gradients
clip_value: float  # clip values exceeding [-clip_value; clip_value]
```

# <a name="own">Building your own agent</a>

If you want to build your own agent, it should always inherit from ```RLAgent```. If your agent uses a replay memory, it should probably inherit from ```MemoryAgent```.

Especially for MemoryAgents, reinforcement learning agents often differ only by their respective value function. In this case, extending the MemoryAgent ist straightforward:

```python
from tensorforce.rl_agents import MemoryAgent
from tensorforce.models import DQNModel

class DQNAgent(MemoryAgent):
	name = 'DQNAgent'

	model_ref = DQNModel
```

Where ```model_ref``` points to the model class. For writing your own RL models, please refer to the [model overview](models.md).