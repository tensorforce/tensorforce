# Update notes

This file records all major updates and new features, starting from version 0.5. As Tensorforce is still developing, updates and bug fixes for the internal architecture are continuously being implemented, which will not be tracked here in detail.


Version 0.5:

New features:
- Automatically configured network type as new default network
- TensorBoard support fully working, for available labels see [summaries test](https://github.com/tensorforce/tensorforce/tree/master/tensorforce/tests/test_summaries.py)
- Many hyperparameters support scheduling and explicit manual re-definition (`tensorforce.core.parameters`)
- New argument `query` for `Agent.act()` and `Agent.observe()`
- Networks and layers functionality improved and extended
- Keras layer support
- Default optimizer plus meta-optimizer features via, for instance, `dict(optimizer='adam', learning_rate=1e-3, multi_step=10, subsampling_fraction=0.2, clipping_value=1e-2, optimized_iterations=5)`

Environment:
- Environment properties `states` and `actions` are now functions `states()` and `actions()`
- See also state/action specification changes below

Agent:
- Agents need to be initialized via `agent.initialize()` before application
- States/actions of type `int` require a parameter `num_values` (instead of `num_actions`)
- `execution` parameter `num_parallel` replaced by a separate argument `parallel_interactions`
- `batched_observe` and `batching_capacity` replaced by argument `buffer_observe`
- `Agent.save/restore_model()` renamed to `Agent.save/restore()`

Networks and layers:
- Directories restructured to `core/networks/` and `core/layers/`
- `LayerBasedNetwork` renamed to `LayerbasedNetwork`
- `Input` layer renamed to `Retrieve` and argument `names` renamed to `tensors`
- `Output` layer renamed to `Register` and argument `name` renamed to `tensor`
- `Nonlinearity` layer renamed to `Activation`
- `Flatten` layer replaced by `Pooling` layer with argument `reduction='concat'`
- `Embedding` layer argument `indices` removed
- `TFLayer`, `Dueling`, `Pool2d` removed (for now)
- Additional changes and extensions of API, see [code](https://github.com/tensorforce/tensorforce/tree/master/tensorforce/core/layers)

Exploration:
- Classes replaced by new parameter classes in `tensorforce.core.parameters`

Runners:
- Improved `run()` API for `Runner` and `ParallelRunner`

Others:
- Baseline option `custom` renamed to `network`
- `TensorForceError` renamed to `TensorforceError`
