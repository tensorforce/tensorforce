Module specification
====================

Agents are instantiated via `Agent.create(agent=...)`, with either of the specification alternatives presented below (`agent` acts as `type` argument). It is recommended to pass as second argument `environment` the application `Environment` implementation, which automatically extracts the corresponding `states`, `actions` and `max_episode_timesteps` arguments of the agent.



### How to specify modules

##### Dictionary with module type and arguments
```python
Agent.create(...
    policy=dict(network=dict(type='layered', layers=[dict(type='dense', size=32)])),
    memory=dict(type='replay', capacity=10000), ...
)
```


##### JSON specification file (plus additional arguments)
```python
Agent.create(...
    policy=dict(network='network.json'),
    memory=dict(type='memory.json', capacity=10000), ...
)
```


##### Module path (plus additional arguments)
```python
Agent.create(...
    policy=dict(network='my_module.TestNetwork'),
    memory=dict(type='tensorforce.core.memories.Replay', capacity=10000), ...
)
```


##### Callable or Type (plus additional arguments)
```python
Agent.create(...
    policy=dict(network=TestNetwork),
    memory=dict(type=Replay, capacity=10000), ...
)
```


##### Default module: only arguments or first argument
```python
Agent.create(...
    policy=dict(network=[dict(type='dense', size=32)]),
    memory=dict(capacity=10000), ...
)
```



### Static vs dynamic hyperparameters

Tensorforce distinguishes between agent/module arguments (primitive types: bool/int/long/float) which specify either part of the TensorFlow model architecture, like the layer size, or a value within the architecture, like the learning rate. Whereas the former are statically defined as part of the agent initialization, the latter can be dynamically adjusted afterwards. These dynamic hyperparameters are indicated by `parameter` as part of their type specification in the documentation, and can alternatively be assigned a [parameter module](../modules/parameters.html) instead of a constant value, for instance, to specify a decaying learning rate.


##### Example: exponentially decaying exploration
```python
Agent.create(...
    exploration=dict(
        type='decaying', unit='timesteps', decay='exponential',
        initial_value=0.1, decay_steps=1000, decay_rate=0.5
    ), ...
)
```


##### Example: linearly increasing horizon
```python
Agent.create(...
    reward_estimation=dict(horizon=dict(
        type='decaying', dtype='long', unit='episodes', decay='polynomial',
        initial_value=10.0, decay_steps=1000, final_value=50.0, power=1.0
    ), ...
)
```
