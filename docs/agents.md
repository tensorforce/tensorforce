# Agent overview

## RLAgent

```RLAgent``` is the base class for all reinforcement learning agents. Every agent inherits from this class.

## MemoryAgent

The ```MemoryAgent``` class implements a replay memory, from which it samples batches to update the value function.

Parameters:

	# Parameters for MemoryAgent
	batch_size: int
	update_rate: float 
	target_network_update_rate: float
	min_replay_size: int
	deterministic_mode: boolean
	use_target_network: boolean
	update_repeat: int

## DQNAgent

Standard DQN agent.

Parameters:

	# Parameters for MemoryAgent
	batch_size: int
	update_rate: float 
	target_network_update_rate: float
	min_replay_size: int
	deterministic_mode: boolean
	use_target_network: boolean
	update_repeat: int
	
	# Parameters for DQNAgent
	-
	
	# Parameters for DQNModel
	double_dqn: boolean
	tau: float
	epsilon: float
	epsilon_final: float
	epsilon_states: int
	gamma: float
	alpha: float
	clip_gradients: boolean
	clip_value: float

# Building your own agent

If you want to build your own agent, it should always inherit from ```RLAgent```. If your agent uses a replay memory, it should probably inherit from ```MemoryAgent```.

Especially for MemoryAgents, reinforcement learning agents often differ only by their respective value function. In this case, extending the MemoryAgent ist straightforward:

	from tensorforce.rl_agents import MemoryAgent
	from tensorforce.models import DQNModel

	class DQNAgent(MemoryAgent):
		name = 'DQNAgent'

		model_ref = DQNModel

Where ```model_ref``` points to the model class. For writing your own RL models, please refer to the [model overview](models.md).