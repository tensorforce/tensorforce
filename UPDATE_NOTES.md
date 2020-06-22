# Update notes

This file records all major updates and new features, starting from version 0.5. As Tensorforce is still developing, updates and bug fixes for the internal architecture are continuously being implemented, which will not be tracked here in detail.



### Version 0.6

- Removed agent arguments: `execution`, `buffer_observe`, `seed` (see next point)
- Removed agent `act()` argument: `deterministic`, `evaluation` (use `independent=True` instead)
- Removed agent function arguments: `query` (functionality removed)
- New agent argument `config` with values: `buffer_observe`, `enable_int_action_masking`, `seed`
- Agent saver functionality changed (Checkpoint/SavedModel instead of Saver/Protobuf): `save`/`load` functions and `saver` argument changed
- Agent summarizer functionality changed: `summarizer` argument changed, some summary labels and other options removed
- Default behavior when specifying `saver` is not to load agent, unless agent is created via `Agent.load`
- Renamed PPO/TRPO/DPG argument: `critic_network/optimizer` to `baseline_network/optimizer`
- Renamed PPO argument: `optimization_steps` to `multi_step`
- Renamed RNN layers: `internal_{rnn/lstm/gru}` to `rnn/lstm/gru`, `rnn/lstm/gru` --> `input_{rnn/lstm/gru}`
- Renamed reward estimation arguments: `estimate_horizon` to `predict_horizon_values`, `estimate_actions` to `predict_action_values`, `estimate_terminal` to `predict_terminal_values`
- Renamed `auto` network argument: `internal_rnn` to `rnn`
- Renamed `(internal_)rnn/lstm/gru` layer argument: `length` to `horizon`
- Renamed `optimizing_step` to `linesearch_step`, and `UpdateModifierWrapper` argument `optimizing_iterations` to `linesearch_iterations`
- Changed default memory `device` argument: `CPU:0`
- Renamed rewards summaries
- Combined `long` and `int` type
- Added double DQN agent (`double_dqn`)
- Always wrap environment in `EnvironmentWrapper` class



### Version 0.5.5

- Changed independent mode of `agent.act` to use final values of dynamic hyperparameters and avoid TensorFlow conditions
- Extended `"tensorflow"` format of `agent.save` to include an optimized Protobuf model with an act-only graph as `.pb` file, and `Agent.load` format `"pb-actonly"` to load act-only agent based on Protobuf model
- Support for custom summaries via new `summarizer` argument value `custom` to specify summary type, and `Agent.summarize(...)` to record summary values
- Added min/max-bounds for dynamic hyperparameters min/max-bounds to assert valid range and infer other arguments
- Argument `batch_size` now mandatory for all agent classes
- Removed `Estimator` argument `capacity`, now always automatically inferred
- Internal changes related to agent arguments `memory`, `update` and `reward_estimation`
- Changed the default `bias` and `activation` argument of some layers
- Fixed issues with `sequence` preprocessor
- DQN and dueling DQN properly constrained to `int` actions only
- Added `use_beta_distribution` argument with default `True` to many agents and `ParametrizedDistributions` policy, so default can be changed



### Version 0.5.4

- DQN/DuelingDQN/DPG argument `memory` now required to be specified explicitly, plus `update_frequency` default changed
- Removed (temporarily) `conv1d/conv2d_transpose` layers due to TensorFlow gradient problems
- `Agent`, `Environment` and `Runner` can now be imported via `from tensorforce import ...`
- New generic reshape layer available as `reshape`
- Support for batched version of `Agent.act` and `Agent.observe`
- Support for parallelized remote environments based on Python's `multiprocessing` and `socket` (replacing `tensorforce/contrib/socket_remote_env/` and `tensorforce/environments/environment_process_wrapper.py`), available via `Environment.create(...)`, `Runner(...)` and `run.py`
- Removed `ParallelRunner` and merged functionality with `Runner`
- Changed `run.py` arguments
- Changed independent mode for `Agent.act`: additional argument `internals` and corresponding return value, initial internals via `Agent.initial_internals()`, `Agent.reset()` not required anymore
- Removed `deterministic` argument for `Agent.act` unless independent mode
- Added `format` argument to `save`/`load`/`restore` with supported formats `tensorflow`, `numpy` and `hdf5`
- Changed `save` argument `append_timestep` to `append` with default `None` (instead of `'timesteps'`)
- Added `get_variable` and `assign_variable` agent functions



### Version 0.5.3

- Added optional `memory` argument to various agents
- Improved summary labels, particularly `"entropy"` and `"kl-divergence"`
- `linear` layer now accepts tensors of rank 1 to 3
- Network output / distribution input does not need to be a vector anymore
- Transposed convolution layers (`conv1d/2d_transpose`)
- Parallel execution functionality contributed by @jerabaul29, currently under `tensorforce/contrib/`
- Accept string for runner `save_best_agent` argument to specify best model directory different from `saver` configuration
- `saver` argument `steps` removed and `seconds` renamed to `frequency`
- Moved `Parallel/Runner` argument `max_episode_timesteps` from `run(...)` to constructor
- New `Environment.create(...)` argument `max_episode_timesteps`
- TensorFlow 2.0 support
- Improved Tensorboard summaries recording
- Summary labels `graph`, `variables` and `variables-histogram` temporarily not working
- TF-optimizers updated to TensorFlow 2.0 Keras optimizers
- Added TensorFlow Addons dependency, and support for TFA optimizers
- Changed unit of `target_sync_frequency` from timesteps to updates for `dqn` and `dueling_dqn` agent



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
