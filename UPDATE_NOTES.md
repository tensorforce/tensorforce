# Update notes

This file records all major updates and new features, starting from version 0.5. As Tensorforce is still developing, updates and bug fixes for the internal architecture are continuously being implemented, which will not be tracked here in detail.



### Latest



### Version 0.5.2

- Improved unittest performance
- Added `updates` and renamed `timesteps`/`episodes` counter for agents and runners
- Renamed `critic_{network,optimizer}` argument to `baseline_{network,optimizer}`
- Added Actor-Critic (`ac`), Advantage Actor-Critic (`a2c`) and Dueling DQN (`dueling_dqn`) agents
- Improved "same" baseline optimizer mode and added optional weight specification
- Reuse layer now global for parameter sharing across modules
- New block layer type (`block`) for easier sharing of layer blocks
- Renamed `PolicyAgent/-Model` to `TensorforceAgent/-Model`
- New `Agent.load(...)` function, saving includes agent specification
- Removed `PolicyAgent` argument `(baseline-)network`
- Added policy argument `temperature`
- Removed `"same"` and `"equal"` options for `baseline_*` arguments and changed internal baseline handling
- Combined `state/action_value` to `value` objective with argument `value` either `"state"` or `"action"`



### Version 0.5.1

- Fixed setup.py packages value



### Version 0.5.0

##### Agent:

- DQFDAgent removed (temporarily)
- DQNNstepAgent and NAFAgent part of DQNAgent
- Agents need to be initialized via `agent.initialize()` before application
- States/actions of type `int` require an entry `num_values` (instead of `num_actions`)
- `Agent.from_spec()` changed and renamed to `Agent.create()`
- `Agent.act()` argument `fetch_tensors` changed and renamed to `query`, `index` renamed to `parallel`, `buffered` removed
- `Agent.observe()` argument `index` renamed to `parallel`
- `Agent.atomic_observe()` removed
- `Agent.save/restore_model()` renamed to `Agent.save/restore()`

##### Agent arguments:

- `update_mode` renamed to `update`
- `states_preprocessing` and `reward_preprocessing` changed and combined to `preprocessing`
- `actions_exploration` changed and renamed to `exploration`
- `execution` entry `num_parallel` replaced by a separate argument `parallel_interactions`
- `batched_observe` and `batching_capacity` replaced by argument `buffer_observe`
- `scope` renamed to `name`

##### DQNAgent arguments:

- `update_mode` replaced by `batch_size`, `update_frequency` and `start_updating`
- `optimizer` removed, implicitly defined as `'adam'`, `learning_rate` added
- `memory` defines capacity of implicitly defined memory `'replay'`
- `double_q_model` removed (temporarily)

##### Policy gradient agent arguments:

- New mandatory argument `max_episode_timesteps`
- `update_mode` replaced by `batch_size` and `update_frequency`
- `memory` removed
- `baseline_mode` removed
- `baseline` argument changed and renamed to `critic_network`
- `baseline_optimizer` renamed to `critic_optimizer`
- `gae_lambda` removed (temporarily)

##### PPOAgent arguments:

- `step_optimizer` removed, implicitly defined as `'adam'`, `learning_rate` added

##### TRPOAgent arguments:

- `cg_*` and `ls_*` arguments removed

##### VPGAgent arguments:

- `optimizer` removed, implicitly defined as `'adam'`, `learning_rate` added

##### Environment:

- Environment properties `states` and `actions` are now functions `states()` and `actions()`
- States/actions of type `int` require an entry `num_values` (instead of `num_actions`)
- New function `Environment.max_episode_timesteps()`

##### Contrib environments:

- ALE, MazeExp, OpenSim, Gym, Retro, PyGame and ViZDoom moved to `tensorforce.environments`
- Other environment implementations removed (may be upgraded in the future)

##### Runners:

- Improved `run()` API for `Runner` and `ParallelRunner`
- `ThreadedRunner` removed

##### Other:

- `examples` folder (including `configs`) removed, apart from `quickstart.py`
- New `benchmarks` folder to replace parts of old `examples` folder
