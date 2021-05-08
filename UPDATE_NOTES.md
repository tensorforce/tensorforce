# Update notes

This file records all major updates and new features, starting from version 0.5. As Tensorforce is still developing, updates and bug fixes for the internal architecture are continuously being implemented, which will not be tracked here in detail.



### Latest changes

##### Agents:
- Agent argument `update_frequency` / `update[frequency]` now supports float values > 0.0, which specify the update-frequency relative to the batch-size
- Changed default value for argument `update_frequency` from `1.0` to `0.25` for DQN, DoubleDQN, DuelingDQN agents
- New function `Agent.get_architecture()` which returns a string representation of the network layer architecture

##### Modules:
- Improved and simplified module specification, for instance: `network=my_module` instead of `network=my_module.TestNetwork`, or `environment=envs.custom_env` instead of `environment=envs.custom_env.CustomEnvironment` (module file needs to be in the same directory or a sub-directory)

##### Networks:
- `KerasNetwork` argument `model` now supports arbitrary functions as long as they return a `tf.keras.Model`

##### Parameters:
- Support tracking of non-constant parameter values

##### Buxfixes:
- Customized device placement was not applied to most tensors


---


### Version 0.6.3

##### Agents:
- New agent argument `tracking` and corresponding function `tracked_tensors()` to track and retrieve the current value of predefined tensors, similar to `summarizer` for TensorBoard summaries
- New experimental value `trace_decay` and `gae_decay` for Tensorforce agent argument `reward_estimation`, soon for other agent types as well
- New options `"early"` and `"late"` for value `estimate_advantage` of Tensorforce agent argument `reward_estimation`
- Changed default value for `Agent.act()` argument `deterministic` from `False` to `True`

##### Networks:
- New network type `KerasNetwork` (specification key: `keras`) as wrapper for networks specified as Keras model
- Passing a Keras model class/object as policy/network argument is automatically interpreted as `KerasNetwork`

##### Distributions:
- Changed `Gaussian` distribution argument `global_stddev=False` to `stddev_mode='predicted'`
- New `Categorical` distribution argument `temperature_mode=None`

##### Layers:
- New option for `Function` layer argument `function` to pass string function expression with argument "x", e.g. "(x+1.0)/2.0"

##### Summarizer:
- New summary `episode-length` recorded as part of summary label "reward"

##### Environments:
- Support for vectorized parallel environments via new function `Environment.is_vectorizable()` and new argument `num_parallel` for `Environment.reset()`
    - See `tensorforce/environments.cartpole.py` for a vectorizable environment example
    - `Runner` uses vectorized parallelism by default if `num_parallel > 1`, `remote=None` and environment supports vectorization
    - See `examples/act_observe_vectorized.py` for more details on act-observe interaction
- New extended and vectorizable custom CartPole environment via key `custom_cartpole` (work in progress)
- New environment argument `reward_shaping` to provide a simple way to modify/shape rewards of an environment, can be specified either as callable or string function expression

##### run.py script:
- New option for command line arguments `--checkpoints` and `--summaries` to add comma-separated checkpoint/summary filename in addition to directory
- Added episode lengths to logging plot besides episode returns

##### Buxfixes:
- Temporal horizon handling of RNN layers
- Critical bugfix for late horizon value prediction (including DQN variants and DPG agent) in combination with baseline RNN
- GPU problems with scatter operations


---


### Version 0.6.2

##### Buxfixes:
- Critical bugfix for DQN variants and DPG agent


---


### Version 0.6.1

##### Agents:
- Removed default value `"adam"` for Tensorforce agent argument `optimizer` (since default optimizer argument `learning_rate` removed, see below)
- Removed option `"minimum"` for Tensorforce agent argument `memory`, use `None` instead
- Changed default value for `dqn`/`double_dqn`/`dueling_dqn` agent argument `huber_loss` from `0.0` to `None`

##### Layers:
- Removed default value `0.999` for `exponential_normalization` layer argument `decay`
- Added new layer `batch_normalization` (generally should only be used for the agent arguments `reward_processing[return_processing]` and `reward_processing[advantage_processing]`)
- Added `exponential/instance_normalization` layer argument `only_mean` with default `False`
- Added `exponential/instance_normalization` layer argument `min_variance` with default `1e-4`

##### Optimizers:
- Removed default value `1e-3` for optimizer argument `learning_rate`
- Changed default value for optimizer argument `gradient_norm_clipping` from `1.0` to `None` (no gradient clipping)
- Added new optimizer `doublecheck_step` and corresponding argument `doublecheck_update` for optimizer wrapper
- Removed `linesearch_step` optimizer argument `accept_ratio`
- Removed `natural_gradient` optimizer argument `return_improvement_estimate`

##### Saver:
- Added option to specify agent argument `saver` as string, which is interpreted as `saver[directory]` with otherwise default values
- Added default value for agent argument `saver[frequency]` as `10` (save model every 10 updates by default)
- Changed default value of agent argument `saver[max_checkpoints]` from `5` to `10`

##### Summarizer:
- Added option to specify agent argument `summarizer` as string, which is interpreted as `summarizer[directory]` with otherwise default values
- Renamed option of agent argument `summarizer` from `summarizer[labels]` to `summarizer[summaries]` (use of the term "label" due to earlier version, outdated and confusing by now)
- Changed interpretation of agent argument `summarizer[summaries] = "all"` to include only numerical summaries, so all summaries except "graph"
- Changed default value of agent argument `summarizer[summaries]` from `["graph"]` to `"all"`
- Changed default value of agent argument `summarizer[max_summaries]` from `5` to `7` (number of different colors in TensorBoard)
- Added option `summarizer[filename]` to agent argument `summarizer`

##### Recorder:
- Added option to specify agent argument `recorder` as string, which is interpreted as `recorder[directory]` with otherwise default values

##### run.py script:
- Added `--checkpoints`/`--summaries`/`--recordings` command line argument to enable saver/summarizer/recorder agent argument specification separate from core agent configuration

##### Examples:
- Added `save_load_agent.py` example script to illustrate regular agent saving and loading

##### Buxfixes:
- Fixed problem with optimizer argument `gradient_norm_clipping` not being applied correctly
- Fixed problem with `exponential_normalization` layer not updating moving mean and variance correctly
- Fixed problem with `recent` memory for timestep-based updates sometimes sampling invalid memory indices


---


### Version 0.6

- Removed agent arguments `execution`, `buffer_observe`, `seed`
- Renamed agent arguments `baseline_policy`/`baseline_network`/`critic_network` to `baseline`/`critic`
- Renamed agent `reward_estimation` arguments `estimate_horizon` to `predict_horizon_values`, `estimate_actions` to `predict_action_values`, `estimate_terminal` to `predict_terminal_values`
- Renamed agent argument `preprocessing` to `state_preprocessing`
- Default agent preprocessing `linear_normalization`
- Moved agent arguments for reward/return/advantage processing from `preprocessing` to `reward_preprocessing` and `reward_estimation[return_/advantage_processing]`
- New agent argument `config` with values `buffer_observe`, `enable_int_action_masking`, `seed`
- Renamed PPO/TRPO/DPG argument `critic_network`/`_optimizer` to `baseline`/`baseline_optimizer`
- Renamed PPO argument `optimization_steps` to `multi_step`
- New TRPO argument `subsampling_fraction`
- Changed agent argument `use_beta_distribution` default to false
- Added double DQN agent (`double_dqn`)
- Removed `Agent.act()` argument `evaluation`
- Removed agent function arguments `query` (functionality removed)
- Agent saver functionality changed (Checkpoint/SavedModel instead of Saver/Protobuf): `save`/`load` functions and `saver` argument changed
- Default behavior when specifying `saver` is not to load agent, unless agent is created via `Agent.load`
- Agent summarizer functionality changed: `summarizer` argument changed, some summary labels and other options removed
- Renamed RNN layers `internal_{rnn/lstm/gru}` to `rnn/lstm/gru` and `rnn/lstm/gru` to `input_{rnn/lstm/gru}`
- Renamed `auto` network argument `internal_rnn` to `rnn`
- Renamed `(internal_)rnn/lstm/gru` layer argument `length` to `horizon`
- Renamed `update_modifier_wrapper` to `optimizer_wrapper`
- Renamed `optimizing_step` to `linesearch_step`, and `UpdateModifierWrapper` argument `optimizing_iterations` to `linesearch_iterations`
- Optimizer `subsampling_step` accepts both absolute (int) and relative (float) fractions
- Objective `policy_gradient` argument `ratio_based` renamed to `importance_sampling`
- Added objectives `state_value` and `action_value`
- Added `Gaussian` distribution arguments `global_stddev` and `bounded_transform` (for improved bounded action space handling)
- Changed default memory `device` argument to `CPU:0`
- Renamed rewards summaries
- `Agent.create()` accepts act-function as `agent` argument for recording
- Singleton states and actions are now consistently handled as singletons
- Major change to policy handling and defaults, in particular `parametrized_distributions`, new default policies `parametrized_state/action_value`
- Combined `long` and `int` type
- Always wrap environment in `EnvironmentWrapper` class
- Changed `tune.py` arguments


---


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


---


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


---


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


---


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


---


### Version 0.5.1

- Fixed setup.py packages value


---


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
